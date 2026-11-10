// 複数分割の FM-index (ウェーブレット行列) を読み込み、 全部の次数の回数を 分割をまたいで足して返す。
// 使い方:
//   fmeng shardlist.txt ORDERS(カンマ区切り, 例 1,2,3,4,5,6,8) batch queries.bin out.bin [threads]
//        queries.bin: 1 問 = int32 x 24 (文脈 23 語, 古い順, 前の穴埋めは -1) + 正解語 -> 計 24 個。 out.bin: 次数ごとに int64 (N, T, c正解) の 3 つ
//   fmeng shardlist.txt ORDERS serve        (標準入力: "D k t1..tk" -> 次数ごとに "o N T m w c w c ..." / "Q" は batch と同じ)
// shardlist.txt の各行: <bwt.xz のパス> <meta.json のパス>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <omp.h>
using namespace std;
static const int L = 15;
struct WM {
  long n = 0; vector<uint64_t> bits[L]; vector<uint32_t> cum[L]; long zeros[L];
  void build(const vector<uint16_t>& v) {
    n = v.size(); vector<uint16_t> cur = v, nxt(n);
    for (int lv = L - 1; lv >= 0; lv--) {
      size_t W = n / 64 + 1; bits[lv].assign(W, 0); cum[lv].assign(W / 8 + 2, 0); long z = 0;
      for (long i = 0; i < n; i++) { if ((cur[i] >> lv) & 1) bits[lv][i >> 6] |= 1ull << (i & 63); else z++; }
      uint32_t c = 0; for (size_t w = 0; w < W; w++) { if ((w & 7) == 0) cum[lv][w >> 3] = c; c += __builtin_popcountll(bits[lv][w]); }
      zeros[lv] = z; long a = 0, b = z; for (long i = 0; i < n; i++) { if ((cur[i] >> lv) & 1) nxt[b++] = cur[i]; else nxt[a++] = cur[i]; } cur.swap(nxt);
    }
  }
  inline long r1(int lv, long i) const {
    size_t w = i >> 6, blk = w >> 3; long c = cum[lv][blk]; for (size_t j = blk << 3; j < w; j++) c += __builtin_popcountll(bits[lv][j]);
    uint64_t m = (i & 63) ? ((1ull << (i & 63)) - 1) : 0; return c + __builtin_popcountll(bits[lv][w] & m);
  }
  long count(uint16_t c, long l, long r) const {
    for (int lv = L - 1; lv >= 0 && l < r; lv--) { long l1 = r1(lv, l), rr1 = r1(lv, r); if ((c >> lv) & 1) { l = zeros[lv] + l1; r = zeros[lv] + rr1; } else { l -= l1; r -= rr1; } }
    return r - l;
  }
  template <class F> void enumerate(int lv, long l, long r, int prefix, F& f) const {
    if (r <= l) return; if (lv < 0) { f(prefix, r - l); return; } long l1 = r1(lv, l), rr1 = r1(lv, r);
    enumerate(lv - 1, l - l1, r - rr1, prefix, f); enumerate(lv - 1, zeros[lv] + l1, zeros[lv] + rr1, prefix | (1 << lv), f);
  }
};
struct Shard { WM wm; long Z = 0; long first = 0; long n = 0; vector<long> C; };
vector<Shard*> SH; vector<int> ORD; vector<long> UNI; long UNIN = 0; int UNIT = 0;

static long jget(const string& s, const char* key) { size_t p = s.find(string("\"") + key + "\""); if (p == string::npos) return -1; p = s.find(':', p); return atol(s.c_str() + p + 1); }

static void load_shard(string bwtxz, string meta, Shard* S) {
  FILE* f = fopen(meta.c_str(), "r"); char buf[4096]; size_t m = fread(buf, 1, 4095, f); buf[m] = 0; fclose(f); string js(buf);
  long n = jget(js, "n"); S->Z = jget(js, "Z") + 1; S->first = jget(js, "first"); S->n = n;
  vector<uint16_t> B2(n + 1); B2[0] = (uint16_t)S->first; string cmd = "xz -dc '" + bwtxz + "'"; FILE* p = popen(cmd.c_str(), "r"); size_t got = fread(B2.data() + 1, 2, n, p); pclose(p);
  if ((long)got != n) { fprintf(stderr, "読み込み失敗 %s: %zu / %ld\n", bwtxz.c_str(), got, n); exit(1); }
  S->C.assign(32769, 0); for (long i = 1; i <= n; i++) S->C[B2[i] + 1]++; S->C[0] += 0;
  // 文章 T の頻度 = BWT の頻度 (Z の行の 0 を 1 つ減らし、 先頭語を 1 つ足す) : B2 には先頭の端の印 (first) が入っているので 数え直す
  S->C[0 + 1] -= 1; S->C[S->first + 1] += 1;
  for (int c = 0; c < 32768; c++) S->C[c + 1] += S->C[c];
  S->wm.build(B2);
}

static inline long cnt(const Shard* S, uint16_t a, long l, long r) { return S->wm.count(a, l, r) - ((a == 0 && l <= S->Z && S->Z < r) ? 1 : 0); }
static bool range_of(const Shard* S, const int* ctx, int k, long& l, long& r) {   // ctx: 古い順 k 語
  l = 0; r = S->n + 1;
  for (int j = 0; j < k && l < r; j++) { uint16_t a = ctx[j]; long nl = 1 + S->C[a] + cnt(S, a, 0, l), nr = 1 + S->C[a] + cnt(S, a, 0, r); l = nl; r = nr; }
  return l < r;
}
struct Res { long N = 0; long T = 0; long cw = 0; };

static void followers(const Shard* S, long l, long r, unordered_map<int, long>& mp) {
  auto f = [&](int sym, long c) { mp[sym] += c; }; S->wm.enumerate(L - 1, l, r, 0, f);
  if (l <= S->Z && S->Z < r) { auto it = mp.find(0); if (it != mp.end()) { it->second -= 1; if (it->second <= 0) mp.erase(it); } }
}

struct Cache { unordered_map<uint64_t, pair<long, long>> m; };

static uint64_t hctx(const int* ctx, int k) { uint64_t h = 1469598103934665603ull + k; for (int i = 0; i < k; i++) { h ^= (uint64_t)(ctx[i] + 7); h *= 1099511628211ull; } return h; }

// 1 つの次数で: N, T (和集合), c(w)。 w < 0 なら c は 0。 followers_out が非 null なら 続く語の 一覧 (和) も返す
static Res query_order(const int* ctx, int k, int w, Cache* cache, unordered_map<int, long>* fo) {
  Res R;
  if (k == 0) { R.N = UNIN; R.T = UNIT; R.cw = w >= 0 ? UNI[w] : 0; if (fo) for (int s = 0; s < 32768; s++) if (UNI[s]) (*fo)[s] = UNI[s]; return R; }
  vector<pair<long, long>> rg; rg.reserve(SH.size()); vector<int> who; long Ntot = 0;
  for (size_t s = 0; s < SH.size(); s++) { long l, r; if (range_of(SH[s], ctx, k, l, r)) { rg.push_back({l, r}); who.push_back(s); Ntot += r - l; if (w >= 0) R.cw += cnt(SH[s], w, l, r); } }
  R.N = Ntot; if (rg.empty()) return R;
  uint64_t key = 0; bool usec = cache && !fo && k <= 3;
  if (usec) { key = hctx(ctx, k); auto it = cache->m.find(key); if (it != cache->m.end()) { R.T = it->second.second; return R; } }
  unordered_map<int, long> local; unordered_map<int, long>& mp = fo ? *fo : local;
  for (size_t j = 0; j < rg.size(); j++) followers(SH[who[j]], rg[j].first, rg[j].second, mp);
  R.T = mp.size(); if (usec) cache->m[key] = {Ntot, R.T}; return R;
}

int main(int argc, char** argv) {
  vector<pair<string, string>> files; { FILE* f = fopen(argv[1], "r"); char a[2048], b[2048]; while (fscanf(f, "%2047s %2047s", a, b) == 2) files.push_back({a, b}); fclose(f); }
  { string o = argv[2]; size_t p = 0; while (p < o.size()) { size_t q = o.find(',', p); if (q == string::npos) q = o.size(); ORD.push_back(atoi(o.substr(p, q - p).c_str())); p = q + 1; } }
  string mode = argv[3]; int nth = (mode == "batch" && argc > 6) ? atoi(argv[6]) : 8; omp_set_num_threads(nth);
  auto t0 = chrono::steady_clock::now(); SH.resize(files.size()); for (auto& s : SH) s = new Shard();
  { int par = 4; vector<thread> th; mutex mu; size_t next = 0;
    for (int t = 0; t < par; t++) th.emplace_back([&]() { while (true) { size_t i; { lock_guard<mutex> g(mu); if (next >= files.size()) return; i = next++; } load_shard(files[i].first, files[i].second, SH[i]); } });
    for (auto& t : th) t.join(); }
  UNI.assign(32768, 0); for (auto* S : SH) for (int c = 0; c < 32768; c++) UNI[c] += S->C[c + 1] - S->C[c]; UNIN = 0; UNIT = 0; for (int c = 0; c < 32768; c++) { UNIN += UNI[c]; if (UNI[c]) UNIT++; }
  fprintf(stderr, "読み込み %zu 分割, 語 %ld, %.1fs\n", SH.size(), UNIN, chrono::duration<double>(chrono::steady_clock::now() - t0).count());
  int no = ORD.size();
  if (mode == "batch") {
    FILE* f = fopen(argv[4], "rb"); fseek(f, 0, SEEK_END); long nq = ftell(f) / (24 * 4); fseek(f, 0, SEEK_SET); vector<int32_t> Q(nq * 24); if (fread(Q.data(), 4, Q.size(), f) != Q.size()) return 1; fclose(f);
    vector<int64_t> out(nq * no * 3); auto t1 = chrono::steady_clock::now();
#pragma omp parallel
    { Cache cache;
#pragma omp for schedule(dynamic, 16)
      for (long q = 0; q < nq; q++) {
        const int32_t* row = &Q[q * 24]; int w = row[23]; int nctx = 0; while (nctx < 23 && row[nctx] < 0) nctx++; nctx = 23 - nctx;   // 有効な文脈の語数
        for (int oi = 0; oi < no; oi++) { int k = ORD[oi] - 1; Res R; if (k <= nctx) { const int* c = row + 23 - k; vector<int> cv(c, c + k); R = query_order(cv.data(), k, w, &cache, nullptr); } out[(q * no + oi) * 3] = R.N; out[(q * no + oi) * 3 + 1] = R.T; out[(q * no + oi) * 3 + 2] = R.cw; }
      } }
    double tq = chrono::duration<double>(chrono::steady_clock::now() - t1).count(); fprintf(stderr, "%ld 問 x %d 次数: %.1fs = %.2f ms/問\n", nq, no, tq, tq / nq * 1e3);
    FILE* g = fopen(argv[5], "wb"); fwrite(out.data(), 8, out.size(), g); fclose(g); return 0;
  }
  // serve: 標準入力 "D k t1..tk" -> 各次数の 続く語の一覧
  Cache cache; char line[8192];
  while (fgets(line, sizeof line, stdin)) {
    char* p = line; if (*p == 'X') break; if (*p != 'D') continue; p++; int k = strtol(p, &p, 10); vector<int> t(k); for (int i = 0; i < k; i++) t[i] = strtol(p, &p, 10);
    for (int oi = 0; oi < no; oi++) {
      int kk = ORD[oi] - 1; if (kk > k) { printf("o %d 0 0 0\n", ORD[oi]); continue; } unordered_map<int, long> fo; Res R = query_order(t.data() + (k - kk), kk, -1, nullptr, &fo);
      printf("o %d %ld %ld %zu", ORD[oi], R.N, R.T, fo.size()); for (auto& kv : fo) printf(" %d %ld", kv.first, kv.second); printf("\n");
    }
    printf("E\n"); fflush(stdout);
  }
  return 0;
}
