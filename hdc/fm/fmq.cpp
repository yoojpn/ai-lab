// ウェーブレット行列 (平坦なビット, 圧縮なし = 速さと正しさの試験用) + 後ろ向き探索。 R = reverse(T) の BWT 上で、 文脈の範囲 -> (N, 異なる続き語数, 正解語の回数)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <functional>
using namespace std;
static const int L = 15;
struct WM {
  long n; vector<vector<uint64_t>> bits; vector<vector<uint32_t>> cum; long zeros[L];
  void build(const vector<uint16_t>& v) {
    n = v.size(); bits.assign(L, {}); cum.assign(L, {}); vector<uint16_t> cur = v, nxt(n);
    for (int lv = L - 1; lv >= 0; lv--) {
      size_t W = n / 64 + 1; bits[lv].assign(W, 0); cum[lv].assign(W + 1, 0); long z = 0;
      for (long i = 0; i < n; i++) { if ((cur[i] >> lv) & 1) bits[lv][i >> 6] |= 1ull << (i & 63); else z++; }
      for (size_t w = 0; w < W; w++) cum[lv][w + 1] = cum[lv][w] + __builtin_popcountll(bits[lv][w]);
      zeros[lv] = z; long a = 0, b = z; for (long i = 0; i < n; i++) { if ((cur[i] >> lv) & 1) nxt[b++] = cur[i]; else nxt[a++] = cur[i]; } cur.swap(nxt);
    }
  }
  inline long r1(int lv, long i) const { uint64_t m = (i & 63) ? ((1ull << (i & 63)) - 1) : 0; return cum[lv][i >> 6] + __builtin_popcountll(bits[lv][i >> 6] & m); }
  long count(uint16_t c, long l, long r) const {
    for (int lv = L - 1; lv >= 0 && l < r; lv--) { long l1 = r1(lv, l), rr1 = r1(lv, r); if ((c >> lv) & 1) { l = zeros[lv] + l1; r = zeros[lv] + rr1; } else { l -= l1; r -= rr1; } }
    return r - l;
  }
  long distinct(int lv, long l, long r) const {
    if (r <= l) return 0; if (lv < 0) return 1; long l1 = r1(lv, l), rr1 = r1(lv, r);
    return distinct(lv - 1, l - l1, r - rr1) + distinct(lv - 1, zeros[lv] + l1, zeros[lv] + rr1);
  }
};
int main(int argc, char** argv) {
  // fmq T.bin bwt.bin positions.bin out.bin
  FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 2; fseek(f, 0, SEEK_SET); vector<uint16_t> T(n); if (fread(T.data(), 2, n, f) != (size_t)n) return 1; fclose(f);
  vector<uint16_t> B(n); f = fopen(argv[2], "rb"); if (fread(B.data(), 2, n, f) != (size_t)n) return 1; fclose(f);
  vector<long> C(32769, 0); for (long i = 0; i < n; i++) C[T[i] + 1]++; for (int c = 0; c < 32768; c++) C[c + 1] += C[c];
  auto t0 = chrono::steady_clock::now(); vector<uint16_t> B2; B2.push_back(T[0]); B2.insert(B2.end(), B.begin(), B.end()); WM wm; wm.build(B2); double tb = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  f = fopen(argv[3], "rb"); fseek(f, 0, SEEK_END); long np = ftell(f) / 4; fseek(f, 0, SEEK_SET); vector<uint32_t> P(np); if (fread(P.data(), 4, np, f) != (size_t)np) return 1; fclose(f);
  long Z=atol(argv[5])+1; auto cnt=[&](uint16_t a,long l,long r){ return wm.count(a,l,r) - ((a==0 && l<=Z && Z<r)?1:0); };
  const int ORD[7] = {1, 2, 3, 4, 5, 6, 8}; vector<int64_t> out(np * 7 * 3);
  t0 = chrono::steady_clock::now();
  for (long q = 0; q < np; q++) {
    long p = P[q]; uint16_t w = T[p];
    for (int oi = 0; oi < 7; oi++) {
      int k = ORD[oi] - 1; long l = 0, r = n + 1;
      for (int j = 0; j < k && l < r; j++) { uint16_t a = T[p - k + j]; long nl = 1 + C[a] + cnt(a, 0, l), nr = 1 + C[a] + cnt(a, 0, r); l = nl; r = nr; }
      long N = r - l; out[(q * 7 + oi) * 3] = N; out[(q * 7 + oi) * 3 + 1] = wm.distinct(L - 1, l, r) - ((l <= Z && Z < r && cnt(0, l, r) == 0) ? 1 : 0); out[(q * 7 + oi) * 3 + 2] = cnt(w, l, r);
    }
  }
  double tq = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "wavelet build %.1fs; %ld positions x 7 orders: %.2fs = %.3f ms/position\n", tb, np, tq, tq / np * 1e3);
  f = fopen(argv[4], "wb"); fwrite(out.data(), 8, out.size(), f); fclose(f); return 0; }
