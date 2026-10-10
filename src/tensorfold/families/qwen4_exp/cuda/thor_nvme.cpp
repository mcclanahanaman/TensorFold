// Flash Next's PLE n-gram rows read from NVMe through a small row cache: Jetson Thor (sm_110).
// A C ABI for ctypes (thor_nvme.py); no Python per read. Rows are the checkpoint's bytes as stored (NVFP4: the code bytes of
// a row, then its e4m3 block-scale bytes); reads are O_DIRECT, whole pages (4 KiB by default), from Linux AIO submitted by
// the calling thread (decode-sized waves: no thread hand-off) and up to ``helpers`` more threads for prompt-sized bursts, or
// from blocking pread threads where AIO is refused. bf16 values come from the caller's pair table, the same lookup as
// host_table.NVFP4Table._gather_pairs, so a row's bits equal the memory-mapped path's. The cache: set-associative, 16 ways
// a set, a 2-bit use count a slot. With ``probation`` ways (S3-FIFO in each set) a new row enters a probation way (FIFO);
// when its way is needed again, a row used meanwhile moves to a main way (CLOCK over the main ways) and an unused one
// leaves, so a one-pass scan of new rows never displaces rows used twice. ``probation`` 0: CLOCK over all ways, a new
// row entering with ``init`` uses.
// Design decision: a shard with a negative scale offset is a repacked row file (one record of code bytes, then
// scale bytes, a row: ONE read a row, two pages when a record straddles one). Arrival reads (``tfnr_prefetch``): a prompt's
// first rows read ahead of its stage by low-priority threads that yield to every foreground gather, inserted as "ahead"
// rows (kAhead) whose first use counts as their insertion, so the cache policy sees what it would without them; the stage
// joins the job (``tfnr_prefetch_wait``) and reads the rest at full width. Snapshot: the cache dumped and loaded set by set.
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

inline void relax() {
#if defined(__aarch64__)
  asm volatile("yield");
#elif defined(__x86_64__)
  asm volatile("pause");
#endif
}

long sys_io_setup(unsigned nr, aio_context_t* ctx) { return syscall(SYS_io_setup, nr, ctx); }
long sys_io_destroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }
long sys_io_submit(aio_context_t ctx, long n, iocb** v) { return syscall(SYS_io_submit, ctx, n, v); }
long sys_io_getevents(aio_context_t ctx, long lo, long hi, io_event* ev) {
  return syscall(SYS_io_getevents, ctx, lo, hi, ev, nullptr);
}

struct Shard { int fd; int64_t codes, scales, rows; };      // scales < 0: packed records of row bytes from ``codes``

constexpr uint8_t kAhead = 0x80;      // a slot's use byte: read ahead (arrival), not used yet
constexpr size_t kMaxAhead = size_t(1) << 20;

// One read: [off, off + len) of fd into a page buffer, then ``n`` bytes from ``at`` of it to ``dst``.
struct Cmd { int fd; uint32_t len; int64_t off; uint8_t* dst; uint32_t at, n; };

struct Job {
  const Cmd* cmds = nullptr;
  int64_t count = 0;
  std::atomic<int64_t> next{0};
  std::atomic<int> err{0};
};

void fail(Job& j, int e) { int z = 0; j.err.compare_exchange_strong(z, e ? e : EIO); }

bool deliver(const Cmd& c, const uint8_t* buf, long got) {
  if (got < long(c.at + c.n)) return false;            // past the end of the file is fine if our bytes came
  std::memcpy(c.dst, buf + c.at, c.n);
  return true;
}

// A thread's reader: AIO with ``depth`` reads in flight (claiming commands as slots free), or one pread at a time.
struct Reader {
  int depth = 0;
  size_t span = 0;
  bool aio = false;
  aio_context_t ctx = 0;
  uint8_t* bufs = nullptr;
  std::vector<iocb> cbs;
  std::vector<iocb*> ptrs;
  std::vector<io_event> evs;
  std::vector<const Cmd*> owner;
  std::vector<int> free_slots;

  bool init(int depth_, size_t span_, bool want_aio) {
    depth = depth_ < 1 ? 1 : depth_;
    span = span_;
    if (posix_memalign(reinterpret_cast<void**>(&bufs), 4096, size_t(depth + 1) * span) != 0) return false;
    aio = want_aio && sys_io_setup(unsigned(depth), &ctx) == 0;
    cbs.resize(depth); ptrs.resize(depth); evs.resize(depth); owner.resize(depth);
    return true;
  }
  ~Reader() {
    if (aio) sys_io_destroy(ctx);
    free(bufs);
  }

  void run(Job& j) {
    if (!aio) {
      for (;;) {
        int64_t k = j.next.fetch_add(1);
        if (k >= j.count) return;
        const Cmd& c = j.cmds[k];
        long got;
        do { got = pread(c.fd, bufs, c.len, c.off); } while (got < 0 && errno == EINTR);
        if (!deliver(c, bufs, got)) fail(j, got < 0 ? errno : EIO);
      }
    }
    free_slots.clear();
    for (int s = depth - 1; s >= 0; --s) free_slots.push_back(s);
    int inflight = 0;
    bool more = true;
    for (;;) {
      int n = 0;
      while (more && !free_slots.empty()) {
        int64_t k = j.next.fetch_add(1);
        if (k >= j.count) { more = false; break; }
        int s = free_slots.back();
        free_slots.pop_back();
        const Cmd& c = j.cmds[k];
        iocb& cb = cbs[s];
        std::memset(&cb, 0, sizeof cb);
        cb.aio_data = uint64_t(s);
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_fildes = uint32_t(c.fd);
        cb.aio_buf = uint64_t(reinterpret_cast<uintptr_t>(bufs + size_t(s) * span));
        cb.aio_nbytes = c.len;
        cb.aio_offset = c.off;
        owner[s] = &c;
        ptrs[n++] = &cb;
      }
      int sent = 0;
      while (sent < n) {
        long r = sys_io_submit(ctx, n - sent, ptrs.data() + sent);
        if (r > 0) { sent += int(r); continue; }
        if (r < 0 && (errno == EINTR || errno == EAGAIN) && inflight > 0) break;   // reap first, then resubmit
        for (int i = sent; i < n; ++i) {                 // refused: read these here, synchronously
          int s = int(ptrs[i]->aio_data);
          const Cmd& c = *owner[s];
          long got;
          do { got = pread(c.fd, bufs + size_t(s) * span, c.len, c.off); } while (got < 0 && errno == EINTR);
          if (!deliver(c, bufs + size_t(s) * span, got)) fail(j, got < 0 ? errno : EIO);
          free_slots.push_back(s);
        }
        n = sent;
        break;
      }
      for (int i = sent; i < n; ++i) {                   // not submitted (EAGAIN): back to the free list, claimed again
        int s = int(ptrs[i]->aio_data);
        pending_.push_back(owner[s]);
        free_slots.push_back(s);
      }
      inflight += sent;
      if (inflight == 0 && !more && pending_.empty()) return;
      if (inflight == 0 && pending_.empty()) continue;
      if (inflight > 0) {
        long got = sys_io_getevents(ctx, 1, inflight, evs.data());
        if (got < 0) {
          if (errno == EINTR) continue;
          fail(j, errno);
          return;
        }
        for (long e = 0; e < got; ++e) {
          int s = int(evs[e].data);
          if (!deliver(*owner[s], bufs + size_t(s) * span, long(evs[e].res))) fail(j, evs[e].res < 0 ? int(-evs[e].res) : EIO);
          free_slots.push_back(s);
        }
        inflight -= int(got);
      }
      while (!pending_.empty()) {                         // EAGAIN'd commands: read here, into the spare buffer
        const Cmd& c = *pending_.back();
        uint8_t* spare = bufs + size_t(depth) * span;
        pending_.pop_back();
        long got;
        do { got = pread(c.fd, spare, c.len, c.off); } while (got < 0 && errno == EINTR);
        if (!deliver(c, spare, got)) fail(j, got < 0 ? errno : EIO);
      }
    }
  }
  std::vector<const Cmd*> pending_;
};

// Threads that wait for one shared task at a time: ``run(parts, fn)`` calls fn(0..parts-1) on them and the caller.
struct Crew {
  std::vector<std::thread> threads;
  std::mutex m, busy;
  std::condition_variable go, done;
  std::function<void(int)> fn;
  std::atomic<int> next{0};
  int parts = 0, active = 0;
  uint64_t gen = 0;
  bool stop = false;

  void start(int n, std::function<void(int)> setup) {
    for (int i = 0; i < n; ++i) threads.emplace_back([this, i, setup] {
      if (setup) setup(i);
      std::unique_lock<std::mutex> lk(m);
      uint64_t seen = 0;
      for (;;) {
        go.wait(lk, [&] { return stop || gen != seen; });
        if (stop) return;
        seen = gen;
        ++active;
        lk.unlock();
        for (int p; (p = next.fetch_add(1)) < parts;) fn(p);
        lk.lock();
        if (--active == 0) done.notify_all();
      }
    });
  }
  // Runs fn over parts with up to ``width`` crew threads plus the caller; alone when another task holds the crew.
  void run(int nparts, int width, const std::function<void(int)>& f) {
    if (!try_run(nparts, width, f))
      for (int p = 0; p < nparts; ++p) f(p);
  }
  bool try_run(int nparts, int width, const std::function<void(int)>& f) {
    if (nparts <= 1 || width <= 0 || threads.empty() || !busy.try_lock()) return false;
    {
      std::unique_lock<std::mutex> lk(m);
      done.wait(lk, [&] { return active == 0; });     // a late joiner of the last task has left
      fn = f;
      parts = nparts;
      next.store(0);
      ++gen;
    }
    if (width >= int(threads.size())) go.notify_all();
    else for (int i = 0; i < width; ++i) go.notify_one();
    for (int p; (p = next.fetch_add(1)) < nparts;) f(p);
    {
      std::unique_lock<std::mutex> lk(m);
      done.wait(lk, [&] { return active == 0; });
    }
    busy.unlock();
    return true;
  }
  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    go.notify_all();
    for (auto& t : threads) t.join();
    threads.clear();
  }
  ~Crew() { shutdown(); }
};

constexpr int kRecords = 1 << 16;

// An arrival read: a prompt's missing rows, read ahead of its stage; claimed in slices by the arrival threads and by a
// waiter (all the rest at once); the thread that completes it inserts the rows.
struct PJob {
  int64_t ticket = 0;
  std::vector<int64_t> rows;
  std::vector<uint8_t> buf;
  std::vector<Cmd> cmds;
  std::atomic<int64_t> next{0}, done{0};
  std::atomic<int> err{0};
  bool finished = false;              // under pf_m
  int users = 0;                      // arrival threads holding it (under pf_m)
  uint64_t t_done = 0;
};

struct Table {
  std::vector<Shard> shards;
  std::vector<int> fds;
  std::vector<int64_t> starts;
  int64_t rows = 0;
  int cb = 0, sb = 0, rb = 0;          // code bytes, scale bytes, row bytes
  size_t page = 4096;
  bool direct = true, aio = true;
  int depth = 64, per_reader = 128;
  // cache
  uint64_t sets = 0;
  int ways = 16, init = 0, probation = 0;
  uint8_t* region = nullptr;
  size_t region_bytes = 0;
  bool locked = false;
  uint32_t* tags = nullptr;
  uint8_t* uses = nullptr;
  uint8_t* hands = nullptr;
  std::atomic<uint8_t>* spin = nullptr;
  uint8_t* data = nullptr;
  // threads
  Crew readers, cpu;
  std::vector<Reader*> crew_readers;
  std::mutex callers_m;
  std::vector<Reader*> callers;
  // stats
  std::atomic<int64_t> st[24];
  std::atomic<int64_t> rec_at{0};
  std::vector<int64_t> recs;          // [kRecords, 4]: rows, wall ns, read ns, unique misses
  // arrival reads
  std::atomic<int> fg{0};             // foreground gathers running: arrival reads wait for them
  std::mutex pf_m;
  std::condition_variable pf_cv, pf_done;
  std::vector<PJob*> pf_jobs;
  std::vector<std::thread> pf_threads;
  std::atomic<bool> pf_stop{false};
  int64_t pf_ticket = 1;
  int pf_width = 2, pf_wave = 64;
  // background load (benchmarks)
  std::atomic<bool> bg_stop{false};
  std::vector<std::thread> bg;
  std::string error;

  ~Table() {
    bg_stop.store(true);
    for (auto& b : bg) b.join();
    {
      std::lock_guard<std::mutex> lk(pf_m);
      pf_stop.store(true);
    }
    pf_cv.notify_all();
    for (auto& p : pf_threads) p.join();
    for (auto* j : pf_jobs) delete j;
    readers.shutdown();
    cpu.shutdown();
    for (auto* r : crew_readers) delete r;
    for (auto* r : callers) delete r;
    for (int fd : fds) close(fd);
    if (region) { if (locked) munlock(region, region_bytes); munmap(region, region_bytes); }
  }

  uint64_t set_of(int64_t row) const {
    uint64_t x = (uint64_t(row) * 0x9E3779B97F4A7C15ull) >> 32;
    return (x * sets) >> 32;
  }
  void lock(uint64_t s) {
    while (spin[s].exchange(1, std::memory_order_acquire))
      while (spin[s].load(std::memory_order_relaxed)) relax();
  }
  void unlock(uint64_t s) { spin[s].store(0, std::memory_order_release); }

  bool lookup(int64_t row, uint8_t* dst) {
    if (!sets) return false;
    uint64_t s = set_of(row), base = s * ways;
    uint32_t tag = uint32_t(row + 1);
    lock(s);
    for (int w = 0; w < ways; ++w) {
      if (tags[base + w] == tag) {
        uint8_t& u = uses[base + w];
        if (u & kAhead) { u = fresh(); st[14].fetch_add(1, std::memory_order_relaxed); }   // its first use: an insertion
        else if (u < 3) ++u;
        if (dst) std::memcpy(dst, data + (base + w) * rb, rb);
        unlock(s);
        return true;
      }
    }
    unlock(s);
    return false;
  }
  uint8_t fresh() const { return uint8_t(probation > 0 ? 0 : init); }
  bool contains(int64_t row) {
    uint64_t s = set_of(row), base = s * ways;
    uint32_t tag = uint32_t(row + 1);
    bool found = false;
    lock(s);
    for (int w = 0; w < ways && !found; ++w) found = tags[base + w] == tag;
    unlock(s);
    return found;
  }
  // CLOCK over ways [lo, hi) of a set: the first way with no uses left, taking one use from each way passed.
  int clock(uint64_t s, int lo, int hi) {
    uint64_t base = s * ways;
    int h = hands[2 * s] < lo || hands[2 * s] >= hi ? lo : hands[2 * s];
    while (uses[base + h]) {
      uint8_t& u = uses[base + h];
      u = (u & kAhead) ? 0 : uint8_t(u - 1);
      h = h + 1 < hi ? h + 1 : lo;
    }
    hands[2 * s] = uint8_t(h + 1 < hi ? h + 1 : lo);
    return h;
  }
  // Puts a row in its set: an empty way, else (probation) the next probation way, its row moved to a main way first if
  // it was used there, else (no probation) CLOCK over all ways. Whether a row left the cache.
  bool insert(int64_t row, const uint8_t* src, bool ahead = false) {
    if (!sets) return false;
    uint64_t s = set_of(row), base = s * ways;
    uint32_t tag = uint32_t(row + 1);
    bool evicted = false;
    lock(s);
    int way = -1;
    for (int w = 0; w < ways; ++w) {
      if (tags[base + w] == tag) { unlock(s); return false; }
      if (way < 0 && tags[base + w] == 0) way = w;
    }
    uint8_t start = uint8_t(init);
    if (way < 0 && probation > 0) {
      way = hands[2 * s + 1] % probation;                 // the probation FIFO's oldest way
      hands[2 * s + 1] = uint8_t((way + 1) % probation);
      if (uses[base + way] & 3) {                         // used while on probation: into the main ways
        int m = clock(s, probation, ways);
        tags[base + m] = tags[base + way];
        uses[base + m] = 1;
        if (src) std::memcpy(data + (base + m) * rb, data + (base + way) * rb, rb);
      }
      evicted = true;
      start = 0;
    } else if (way < 0) {
      way = clock(s, 0, ways);
      evicted = true;
    }
    tags[base + way] = tag;
    uses[base + way] = ahead ? kAhead : start;
    if (src) std::memcpy(data + (base + way) * rb, src, rb);
    unlock(s);
    return evicted;
  }

  Reader* take_reader() {
    {
      std::lock_guard<std::mutex> lk(callers_m);
      if (!callers.empty()) { Reader* r = callers.back(); callers.pop_back(); return r; }
    }
    Reader* r = new Reader();
    r->init(depth, 2 * page, aio);
    return r;
  }
  void give_reader(Reader* r) { std::lock_guard<std::mutex> lk(callers_m); callers.push_back(r); }

  int read(const Cmd* cmds, int64_t count) {
    if (count <= 0) return 0;
    Job j;
    j.cmds = cmds;
    j.count = count;
    int width = aio ? int((j.count + per_reader - 1) / per_reader) - 1 : int(j.count) - 1;
    if (width > int(readers.threads.size())) width = int(readers.threads.size());
    Reader* mine = take_reader();
    if (width <= 0 || !readers.try_run(width + 1, width, [&](int p) { (p ? crew_readers[p - 1] : mine)->run(j); }))
      mine->run(j);                     // a small wave, or another call holds the crew: this thread reads it all
    give_reader(mine);
    return j.err.load();
  }

  int64_t shard_of(int64_t row) const {
    int64_t lo = 0, hi = int64_t(shards.size()) - 1;
    while (lo < hi) {
      int64_t mid = (lo + hi + 1) / 2;
      if (starts[mid] <= row) lo = mid; else hi = mid - 1;
    }
    return lo;
  }
  void add(std::vector<Cmd>& cmds, int fd, int64_t off, uint32_t n, uint8_t* dst) const {
    int64_t a = off & ~int64_t(page - 1), end = off + n;
    uint32_t len = uint32_t(((end - a) + int64_t(page) - 1) & ~int64_t(page - 1));
    cmds.push_back(Cmd{fd, len, a, dst, uint32_t(off - a), n});
  }
  // A row's reads into dst (code bytes, then scale bytes): one for a packed record, two for the checkpoint as stored.
  void commands(std::vector<Cmd>& cmds, int64_t row, uint8_t* dst) const {
    int64_t f = shard_of(row), local = row - starts[size_t(f)];
    const Shard& s = shards[size_t(f)];
    if (s.scales < 0) { add(cmds, s.fd, s.codes + local * rb, uint32_t(rb), dst); return; }
    add(cmds, s.fd, s.codes + local * cb, uint32_t(cb), dst);
    add(cmds, s.fd, s.scales + local * sb, uint32_t(sb), dst + cb);
  }

  // Arrival reads: up to ``width`` of a job's unclaimed reads (all the rest: width < 0) with reader r (nullptr: this
  // thread and the reader crew); the thread that completes the job inserts its rows (none after an error).
  void claim(PJob& j, Reader* r, int64_t width) {
    int64_t total = int64_t(j.cmds.size());
    int64_t a = j.next.fetch_add(width < 0 ? total : width);
    if (a >= total) return;
    int64_t b = width < 0 ? total : std::min(total, a + width);
    int e;
    if (r) {
      Job w;
      w.cmds = j.cmds.data() + a;
      w.count = b - a;
      r->run(w);
      e = w.err.load();
    } else {
      e = read(j.cmds.data() + a, b - a);
    }
    if (e) { int z = 0; j.err.compare_exchange_strong(z, e); }
    if (j.done.fetch_add(b - a) + (b - a) == total) finish(j);
  }
  void finish(PJob& j) {
    if (!j.err.load()) {
      int64_t ev = 0;
      for (size_t u = 0; u < j.rows.size(); ++u) ev += insert(j.rows[u], j.buf.data() + u * size_t(rb), true);
      st[11].fetch_add(ev);
      st[13].fetch_add(int64_t(j.rows.size()));
    }
    std::lock_guard<std::mutex> lk(pf_m);
    j.finished = true;
    j.t_done = now_ns();
    pf_done.notify_all();
  }
  void arrival_loop() {
    Reader r;
    if (!r.init(pf_wave, 2 * page, aio)) return;
    std::unique_lock<std::mutex> lk(pf_m);
    for (;;) {
      PJob* j = nullptr;
      pf_cv.wait(lk, [&] {
        if (pf_stop.load()) return true;
        for (PJob* q : pf_jobs)
          if (!q->finished && q->next.load() < int64_t(q->cmds.size())) { j = q; return true; }
        return false;
      });
      if (pf_stop.load()) return;
      ++j->users;
      lk.unlock();
      while (fg.load(std::memory_order_relaxed) > 0 && !pf_stop.load())   // foreground (decode, stage) reads first
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      claim(*j, &r, pf_wave);
      lk.lock();
      --j->users;
      pf_done.notify_all();
    }
  }
};

thread_local std::vector<uint8_t> t_stage, t_miss, t_uraw;
thread_local std::vector<int64_t> t_keys, t_urows;
thread_local std::vector<uint32_t> t_vals, t_uidx;
thread_local std::vector<Cmd> t_cmds;

}  // namespace

extern "C" {

const char* tfnr_error(Table* t) { return t ? t->error.c_str() : "no table"; }

// files: paths; shards: per shard (file index, code bytes offset, scale bytes offset, rows) as int64 quadruples.
Table* tfnr_open(int nfiles, const char** paths, int nshards, const int64_t* shard_info, int code_bytes, int scale_bytes,
                 int64_t cache_rows, int direct, int aio, int depth, int helpers, int cpu_threads, int page_bytes,
                 int init_uses, int probation, char* err, int errlen) {
  auto* t = new Table();
  auto bad = [&](const std::string& why) -> Table* {
    std::snprintf(err, size_t(errlen), "%s", why.c_str());
    delete t;
    return nullptr;
  };
  for (auto& s : t->st) s.store(0);
  t->cb = code_bytes; t->sb = scale_bytes; t->rb = code_bytes + scale_bytes;
  t->page = size_t(page_bytes); t->direct = direct != 0; t->aio = aio != 0; t->depth = depth; t->init = init_uses;
  t->probation = probation < 0 ? 0 : probation >= t->ways ? t->ways - 1 : probation;
  if (t->page < 512 || (t->page & (t->page - 1))) return bad("page bytes must be a power of two >= 512");
  for (int i = 0; i < nfiles; ++i) {
    int fd = open(paths[i], O_RDONLY | O_CLOEXEC | (t->direct ? O_DIRECT : 0));
    if (fd < 0) return bad(std::string("open ") + paths[i] + ": " + std::strerror(errno));
    t->fds.push_back(fd);
  }
  t->starts.push_back(0);
  for (int i = 0; i < nshards; ++i) {
    const int64_t* q = shard_info + 4 * i;
    if (q[0] < 0 || q[0] >= nfiles) return bad("shard file index out of range");
    t->shards.push_back(Shard{t->fds[size_t(q[0])], q[1], q[2], q[3]});
    t->starts.push_back(t->starts.back() + q[3]);
  }
  t->rows = t->starts.back();
  if (t->rows >= (int64_t(1) << 32) - 1) return bad("too many rows for 32-bit tags");
  if (cache_rows > 0) {
    t->sets = uint64_t(cache_rows + t->ways - 1) / uint64_t(t->ways);
    uint64_t slots = t->sets * uint64_t(t->ways);
    size_t bytes = slots * size_t(t->rb) + slots * 4 + slots + t->sets * 3 + 4096;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return bad(std::string("cache mmap: ") + std::strerror(errno));
    t->region = static_cast<uint8_t*>(p);
    t->region_bytes = bytes;
    t->locked = mlock(p, bytes) == 0;
    t->data = t->region;
    t->tags = reinterpret_cast<uint32_t*>(t->region + slots * size_t(t->rb));
    t->uses = reinterpret_cast<uint8_t*>(t->tags + slots);
    t->hands = t->uses + slots;
    t->spin = reinterpret_cast<std::atomic<uint8_t>*>(t->hands + 2 * t->sets);
  }
  t->recs.assign(size_t(kRecords) * 4, 0);
  t->per_reader = 2 * (depth < 1 ? 1 : depth);
  {
    Reader probe;                       // AIO refused (seccomp, kernel): every reader reads with pread instead
    probe.init(1, 2 * t->page, t->aio);
    t->aio = probe.aio;
  }
  int nh = t->aio ? helpers : (helpers > 31 ? helpers : 31);   // pread threads keep the queue deep only by number
  for (int i = 0; i < nh; ++i) {
    auto* r = new Reader();
    if (!r->init(depth, 2 * t->page, t->aio)) { delete r; return bad("reader buffers"); }
    t->crew_readers.push_back(r);
  }
  t->readers.start(int(t->crew_readers.size()), nullptr);
  t->cpu.start(cpu_threads, nullptr);
  return t;
}

void tfnr_close(Table* t) { delete t; }

// info: [rows, cache slots, cache bytes, locked, aio, direct, page, reader threads, cpu threads, depth, probation ways]
void tfnr_info(Table* t, int64_t* out) {
  out[0] = t->rows; out[1] = int64_t(t->sets) * t->ways; out[2] = int64_t(t->region_bytes); out[3] = t->locked;
  out[4] = t->aio; out[5] = t->direct; out[6] = int64_t(t->page); out[7] = int64_t(t->crew_readers.size());
  out[8] = int64_t(t->cpu.threads.size()); out[9] = t->depth; out[10] = t->probation;
}

// Rows ``ids`` -> ``out``: kind 0 raw rows [n, row bytes]; kind 1 [n, code bytes] uint32 words from ``pairs``
// (pairs[(scale byte << 8) | code byte] = the code byte's two bf16 values). 0, or a positive errno.
int tfnr_gather(Table* t, const int64_t* ids, int64_t n, const uint32_t* pairs, void* out, int kind) {
  uint64_t t0 = now_ns();
  struct Busy {
    std::atomic<int>& c;
    explicit Busy(std::atomic<int>& x) : c(x) { c.fetch_add(1); }
    ~Busy() { c.fetch_sub(1); }
  } busy(t->fg);
  const int rb = t->rb, cb = t->cb, sb = t->sb;
  if (kind == 1 && (pairs == nullptr || cb != 8 * sb)) return EINVAL;
  for (int64_t i = 0; i < n; ++i)
    if (ids[i] < 0 || ids[i] >= t->rows) { t->error = "row id out of range"; return ERANGE; }
  uint8_t* stage;
  if (kind == 0) stage = static_cast<uint8_t*>(out);
  else { t_stage.resize(size_t(n) * rb); stage = t_stage.data(); }
  t_miss.resize(size_t(n));
  uint8_t* miss = t_miss.data();
  const int64_t piece = 4096;
  int parts = int((n + piece - 1) / piece);
  std::atomic<int64_t> hits{0};
  t->cpu.run(parts, parts - 1, [&](int p) {
    int64_t lo = p * piece, hi = lo + piece < n ? lo + piece : n, h = 0;
    for (int64_t i = lo; i < hi; ++i) {
      bool hit = t->lookup(ids[i], stage + size_t(i) * rb);
      miss[i] = !hit;
      h += hit;
    }
    hits.fetch_add(h);
  });
  int64_t nmiss = n - hits.load();
  int64_t nu = 0;
  uint64_t r0 = now_ns(), r1 = r0;
  int rc = 0;
  if (nmiss > 0) {
    size_t cap = 16;
    while (cap < size_t(2 * nmiss)) cap <<= 1;
    t_keys.assign(cap, -1);
    t_vals.resize(cap);
    t_urows.clear();
    t_uidx.resize(size_t(nmiss));
    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
      if (!miss[i]) continue;
      int64_t row = ids[i];
      size_t hpos = size_t((uint64_t(row) * 0x9E3779B97F4A7C15ull) >> 20) & (cap - 1);
      while (t_keys[hpos] != -1 && t_keys[hpos] != row) hpos = (hpos + 1) & (cap - 1);
      if (t_keys[hpos] == -1) { t_keys[hpos] = row; t_vals[hpos] = uint32_t(t_urows.size()); t_urows.push_back(row); }
      t_uidx[size_t(k++)] = t_vals[hpos];
    }
    nu = int64_t(t_urows.size());
    t_uraw.resize(size_t(nu) * rb);
    t_cmds.clear();
    t_cmds.reserve(size_t(2 * nu));
    for (int64_t u = 0; u < nu; ++u) t->commands(t_cmds, t_urows[size_t(u)], t_uraw.data() + size_t(u) * rb);
    r0 = now_ns();
    rc = t->read(t_cmds.data(), int64_t(t_cmds.size()));
    r1 = now_ns();
    if (rc) { t->st[9].fetch_add(1); t->error = std::string("n-gram row read: ") + std::strerror(rc); return rc; }
    int64_t ev = 0;
    for (int64_t u = 0; u < nu; ++u) ev += t->insert(t_urows[size_t(u)], t_uraw.data() + size_t(u) * rb);
    k = 0;
    for (int64_t i = 0; i < n; ++i)
      if (miss[i]) std::memcpy(stage + size_t(i) * rb, t_uraw.data() + size_t(t_uidx[size_t(k++)]) * rb, size_t(rb));
    t->st[10].fetch_add(nu);
    t->st[11].fetch_add(ev);
    t->st[5].fetch_add(int64_t(t_cmds.size()));
    int64_t bytes = 0;
    for (const Cmd& c : t_cmds) bytes += c.len;
    t->st[6].fetch_add(bytes);
  }
  if (kind == 1) {
    uint32_t* words = static_cast<uint32_t*>(out);
    t->cpu.run(parts, parts - 1, [&](int p) {
      int64_t lo = p * piece, hi = lo + piece < n ? lo + piece : n;
      for (int64_t i = lo; i < hi; ++i) {
        const uint8_t* codes = stage + size_t(i) * rb;
        const uint8_t* scales = codes + cb;
        uint32_t* o = words + size_t(i) * cb;
        for (int b = 0; b < sb; ++b) {
          const uint32_t* pr = pairs + (uint32_t(scales[b]) << 8);
          for (int c = 0; c < 8; ++c) o[b * 8 + c] = pr[codes[b * 8 + c]];
        }
      }
    });
  }
  uint64_t t1 = now_ns();
  t->st[0].fetch_add(1); t->st[1].fetch_add(n); t->st[2].fetch_add(n - nmiss); t->st[3].fetch_add(nmiss);
  t->st[4].fetch_add(nu); t->st[7].fetch_add(int64_t(r1 - r0)); t->st[8].fetch_add(int64_t(t1 - t0));
  int64_t at = t->rec_at.fetch_add(1) & (kRecords - 1);
  int64_t* r = t->recs.data() + at * 4;
  r[0] = n; r[1] = int64_t(t1 - t0); r[2] = int64_t(r1 - r0); r[3] = nu;
  return 0;
}

// stats: [calls, lookups, hits, missed lookups, unique misses, reads, bytes read, read ns, gather ns, errors, inserted,
// evicted, ...]; records: up to ``max`` of the last calls [rows, wall ns, read ns, unique misses]; the count.
void tfnr_stats(Table* t, int64_t* out) { for (int i = 0; i < 24; ++i) out[i] = t->st[i].load(); }
int64_t tfnr_records(Table* t, int64_t* out, int64_t max) {
  int64_t total = t->rec_at.load(), n = total < kRecords ? total : kRecords;
  if (n > max) n = max;
  for (int64_t k = 0; k < n; ++k) {
    int64_t at = (total - n + k) & (kRecords - 1);
    std::memcpy(out + 4 * k, t->recs.data() + at * 4, 4 * sizeof(int64_t));
  }
  return n;
}
void tfnr_reset_stats(Table* t) { for (auto& s : t->st) s.store(0); t->rec_at.store(0); }

// Drops every cached row (a cold start for measurements); safe beside gathers (set by set under its lock).
void tfnr_drop(Table* t) {
  for (uint64_t s = 0; s < t->sets; ++s) {
    t->lock(s);
    for (int w = 0; w < t->ways; ++w) { t->tags[s * t->ways + w] = 0; t->uses[s * t->ways + w] = 0; }
    t->hands[2 * s] = t->hands[2 * s + 1] = 0;
    t->unlock(s);
  }
}

// The cache policy alone (no reads, no bytes): each id looked up, inserted on a miss; hit[i] = 1 on a hit.
void tfnr_simulate(Table* t, const int64_t* ids, int64_t n, uint8_t* hit) {
  for (int64_t i = 0; i < n; ++i) {
    bool h = t->lookup(ids[i], nullptr);
    if (!h) t->insert(ids[i], nullptr);
    if (hit) hit[i] = h;
  }
}

// Benchmarks: ``threads`` background readers of random rows' code pages until stopped (0 stops them).
void tfnr_background(Table* t, int threads) {
  t->bg_stop.store(true);
  for (auto& b : t->bg) b.join();
  t->bg.clear();
  t->bg_stop.store(false);
  for (int i = 0; i < threads; ++i) t->bg.emplace_back([t, i] {
    uint8_t* buf = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&buf), 4096, 2 * t->page) != 0) return;
    uint64_t x = 0x9E3779B97F4A7C15ull * uint64_t(i + 1);
    while (!t->bg_stop.load(std::memory_order_relaxed)) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      int64_t row = int64_t(x % uint64_t(t->rows)), f = t->shard_of(row);
      const Shard& s = t->shards[size_t(f)];
      int64_t off = (s.codes + (row - t->starts[size_t(f)]) * (s.scales < 0 ? t->rb : t->cb)) & ~int64_t(t->page - 1);

      if (pread(s.fd, buf, t->page, off) < 0 && errno != EINTR) break;
    }
    free(buf);
  });
}

// Arrival reads: ``threads`` low-priority readers (started at the first job), ``wave`` reads each in flight.
void tfnr_arrival(Table* t, int threads, int wave) {
  std::lock_guard<std::mutex> lk(t->pf_m);
  if (t->pf_threads.empty()) {
    t->pf_width = threads < 1 ? 1 : threads;
    t->pf_wave = wave < 1 ? 1 : wave;
  }
}

// Starts reading the rows ``ids`` that are not cached (unique, at most 2^20) ahead of their gather: a ticket for
// ``tfnr_prefetch_wait``, or 0 when nothing needs reading (no cache, all cached).
int64_t tfnr_prefetch(Table* t, const int64_t* ids, int64_t n) {
  if (!t->sets || n <= 0) return 0;
  std::vector<int64_t> rows(ids, ids + n);
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  auto* j = new PJob();
  for (int64_t r : rows)
    if (r >= 0 && r < t->rows && !t->contains(r) && j->rows.size() < kMaxAhead) j->rows.push_back(r);
  t->st[12].fetch_add(n);
  if (j->rows.empty()) { delete j; return 0; }
  j->buf.resize(j->rows.size() * size_t(t->rb));
  j->cmds.reserve(2 * j->rows.size());
  for (size_t u = 0; u < j->rows.size(); ++u) t->commands(j->cmds, j->rows[u], j->buf.data() + u * size_t(t->rb));
  std::lock_guard<std::mutex> lk(t->pf_m);
  if (t->pf_threads.empty())
    for (int i = 0; i < t->pf_width; ++i) t->pf_threads.emplace_back([t] { t->arrival_loop(); });
  uint64_t now = now_ns();                 // jobs never waited for (a refused request): dropped a minute after they end
  std::vector<PJob*> keep;
  for (PJob* q : t->pf_jobs) {
    if (q->finished && q->users == 0 && now - q->t_done > 60000000000ull) delete q;
    else keep.push_back(q);
  }
  t->pf_jobs.swap(keep);

  j->ticket = t->pf_ticket++;
  t->pf_jobs.push_back(j);
  t->st[16].fetch_add(1);
  t->pf_cv.notify_all();
  return j->ticket;
}

// Waits for an arrival job, reading all its unclaimed rows at full width first (``help`` 0: drops them unread, and the
// job's rows are not inserted); 0, or the positive errno of a failed read (its rows are then not cached).
int tfnr_prefetch_wait(Table* t, int64_t ticket, int help) {
  if (ticket <= 0) return 0;
  PJob* j = nullptr;
  {
    std::lock_guard<std::mutex> lk(t->pf_m);
    for (PJob* q : t->pf_jobs) if (q->ticket == ticket) { j = q; break; }
    if (j) ++j->users;
  }
  if (!j) return 0;
  uint64_t w0 = now_ns();
  if (help) {
    t->claim(*j, nullptr, -1);
  } else {
    int64_t total = int64_t(j->cmds.size()), a = j->next.fetch_add(total);
    if (a < total) {
      int z = 0;
      j->err.compare_exchange_strong(z, ECANCELED);
      if (j->done.fetch_add(total - a) + (total - a) == total) t->finish(*j);
    }
  }
  std::unique_lock<std::mutex> lk(t->pf_m);
  --j->users;
  t->pf_done.wait(lk, [&] { return j->finished && j->users == 0; });
  int e = j->err.load();
  t->pf_jobs.erase(std::remove(t->pf_jobs.begin(), t->pf_jobs.end(), j), t->pf_jobs.end());
  delete j;
  t->st[15].fetch_add(int64_t(now_ns() - w0));
  return e == ECANCELED ? 0 : e;
}

// Snapshot (warm start): [tags u32 x slots][uses u8 x slots][hands u8 x 2 sets][row bytes x slots], copied set by set
// under each set's lock; a row read ahead and not used yet is saved as unused. The bytes; the dump returns the rows held.
int64_t tfnr_snapshot_bytes(Table* t) {
  uint64_t slots = t->sets * uint64_t(t->ways);
  return int64_t(slots * (5 + uint64_t(t->rb)) + 2 * t->sets);
}
int64_t tfnr_snapshot_dump(Table* t, uint8_t* out) {
  const uint64_t slots = t->sets * uint64_t(t->ways), W = uint64_t(t->ways), rb = uint64_t(t->rb);
  uint8_t* tags = out;
  uint8_t* uses = out + 4 * slots;
  uint8_t* hands = uses + slots;
  uint8_t* data = hands + 2 * t->sets;
  int64_t held = 0;
  for (uint64_t s = 0; s < t->sets; ++s) {
    const uint64_t b = s * W;
    t->lock(s);
    std::memcpy(tags + 4 * b, t->tags + b, 4 * W);
    for (uint64_t w = 0; w < W; ++w) uses[b + w] = t->uses[b + w] & 3;
    hands[2 * s] = t->hands[2 * s];
    hands[2 * s + 1] = t->hands[2 * s + 1];
    std::memcpy(data + b * rb, t->data + b * rb, W * rb);
    t->unlock(s);
    for (uint64_t w = 0; w < W; ++w) held += t->tags[b + w] != 0;
  }
  return held;
}
// Loads a dump of this geometry (sets, ways, row bytes). Every slot is checked first (a tag in its own set and in range,
// use counts and hands in range): EINVAL and nothing loaded on any mismatch; 0 after the load.
int tfnr_snapshot_load(Table* t, const uint8_t* in, int64_t sets, int ways, int row_bytes) {
  if (!t->sets || sets < 0 || uint64_t(sets) != t->sets || ways != t->ways || row_bytes != t->rb) return EINVAL;
  const uint64_t slots = t->sets * uint64_t(t->ways), W = uint64_t(t->ways), rb = uint64_t(t->rb);
  const uint8_t* tags = in;
  const uint8_t* uses = in + 4 * slots;
  const uint8_t* hands = uses + slots;
  const uint8_t* data = hands + 2 * t->sets;
  const int pw = t->probation > 0 ? t->probation : 1;
  for (uint64_t s = 0; s < t->sets; ++s) {
    if (hands[2 * s] >= W || hands[2 * s + 1] >= uint64_t(pw)) return EINVAL;
    for (uint64_t w = 0; w < W; ++w) {
      uint32_t tag;
      std::memcpy(&tag, tags + 4 * (s * W + w), 4);
      if (uses[s * W + w] > 3) return EINVAL;
      if (tag && (int64_t(tag) - 1 >= t->rows || t->set_of(int64_t(tag) - 1) != s)) return EINVAL;
    }
  }
  for (uint64_t s = 0; s < t->sets; ++s) {
    const uint64_t b = s * W;
    t->lock(s);
    std::memcpy(t->tags + b, tags + 4 * b, 4 * W);
    std::memcpy(t->uses + b, uses + b, W);
    t->hands[2 * s] = hands[2 * s];
    t->hands[2 * s + 1] = hands[2 * s + 1];
    std::memcpy(t->data + b * rb, data + b * rb, W * rb);
    t->unlock(s);
  }
  return 0;
}

}  // extern "C"

