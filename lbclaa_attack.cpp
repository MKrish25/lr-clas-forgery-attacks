#include "lattice.hpp"
#include <cinttypes>

struct Params {
  size_t n = 512;
  size_t k = 256;
  size_t m = 6144;
  int64  q = 8380417;
  int64  a = 100;
  int64  theta = 5000;
  double sigma = 15000;
  double s = 2000;
  int    dh = 4;
};

struct Vehicle {
  Mat chi;
  Mat S1;
  Mat ID1;
  Mat U;
  std::string pid;
};

struct SingleSig {
  std::vector<int64> eta1;
  std::vector<int64> eta2;
  std::vector<int64> d1;
  std::vector<int64> d2;
  std::vector<int64> h;
};

struct AggSig {
  std::vector<int64> eta_agg1, eta_agg2, d_agg1, d_agg2;
  std::vector<std::vector<int64>> hs;
};

static Mat random_mat(size_t r, size_t c, int64 q, Xoshiro256& rng) {
  Mat M(r, c, q);
  for (auto& v : M.v) v = rng.uniform_int(-q / 2, q / 2);
  return M;
}



static std::vector<int64> sample_challenge(size_t k, int dh, Xoshiro256& rng) {
  std::vector<int64> h(k, 0);
  std::vector<size_t> idx(k);
  for (size_t i = 0; i < k; ++i) idx[i] = i;
  std::shuffle(idx.begin(), idx.end(), std::mt19937(rng.next()));
  for (int t = 0; t < dh; ++t)
    h[idx[t]] = (rng.next() & 1) ? 1 : -1;
  return h;
}

static std::vector<int64> mat_short_mul(const Mat& M, const std::vector<int64>& h,
                                        int64 q) {
  std::vector<int64> out(M.rows, 0);
  for (size_t i = 0; i < M.rows; ++i) {
    __int128 acc = 0;
    for (size_t j = 0; j < h.size(); ++j) acc += (__int128)M(i, j) * h[j];
    out[i] = fmodq((int64)(acc % q), q);
  }
  return out;
}

int main(int argc, char** argv) {
  Params P;
  int vehicles = 8;
  unsigned long long seed = 0xD11CE2026ULL;
  bool small = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--small") small = true;
    else if (a == "--vehicles" && i + 1 < argc) vehicles = atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 0);
  }
  if (small) P.n = 64, P.k = 64, P.m = 768, P.dh = 3;

  Xoshiro256 rng(seed);
  size_t n = P.n, k = P.k, m = P.m;
  int64 q = P.q;

  Mat A = random_mat(n, m, q, rng);
  Mat A1 = random_mat(n, m, q, rng);
  double beta = 2.0 * P.sigma * std::sqrt((double)m);

  Vehicle v;
  v.chi = Mat(m, k, q);
  for (auto& x : v.chi.v) x = rng.uniform_int(-P.a, P.a);
  v.S1 = Mat(m, k, q);
  for (auto& x : v.S1.v) x = (int64)std::llround(snorm_round(rng) * P.s);
  v.ID1 = mat_mul(A1, v.chi, q);
  v.U = mat_mul(A, v.S1, q);
  v.pid = "PID-0";

  SingleSig sig;

  int trials = 0;
  for (;;) {
    ++trials;
    std::vector<int64> y1 = gauss_vec(m, P.sigma, rng);
    std::vector<int64> y2 = gauss_vec(m, P.sigma, rng);
    sig.eta1 = mat_vec(A1, y1, q);
    sig.eta2 = mat_vec(A, y2, q);
    std::vector<int64> hraw = sample_challenge(k, P.dh, rng);
    sig.h = hraw;
    sig.d1 = mat_short_mul(v.chi, hraw, q);
    sig.d2 = mat_short_mul(v.S1, hraw, q);
    sig.d1 = vec_add(sig.d1, y1, q);
    sig.d2 = vec_add(sig.d2, y2, q);
    if (l2(sig.d1) <= beta && l2(sig.d2) <= beta) break;
  }

  bool n1 = (l2(sig.d1) <= beta), n2 = (l2(sig.d2) <= beta);
  std::vector<int64> r1 = vec_sub(mat_vec(A1, sig.d1, q),
                                  mat_short_mul(v.ID1, sig.h, q), q);
  std::vector<int64> r2 = vec_sub(mat_vec(A, sig.d2, q),
                                  mat_short_mul(v.U, sig.h, q), q);
  bool e1 = (r1 == sig.eta1), e2 = (r2 == sig.eta2);
  bool honest_single = n1 && n2 && e1 && e2;


  std::vector<Vehicle> vehs;
  vehs.push_back(v);
  std::vector<SingleSig> sigs;
  sigs.push_back(sig);
  for (int i = 1; i < vehicles; ++i) {
    Vehicle w;
    w.chi = Mat(m, k, q);
    for (auto& x : w.chi.v) x = rng.uniform_int(-P.a, P.a);
    w.S1 = Mat(m, k, q);
    for (auto& x : w.S1.v) x = (int64)std::llround(snorm_round(rng) * P.s);
    w.ID1 = mat_mul(A1, w.chi, q);
    w.U = mat_mul(A, w.S1, q);
    w.pid = "PID-" + std::to_string(i);
    vehs.push_back(w);
    SingleSig ss;
    for (;;) {
      std::vector<int64> y1 = gauss_vec(m, P.sigma, rng);
      std::vector<int64> y2 = gauss_vec(m, P.sigma, rng);
      ss.eta1 = mat_vec(A1, y1, q);
      ss.eta2 = mat_vec(A, y2, q);
      ss.h = sample_challenge(k, P.dh, rng);
      ss.d1 = vec_add(mat_short_mul(w.chi, ss.h, q), y1, q);
      ss.d2 = vec_add(mat_short_mul(w.S1, ss.h, q), y2, q);
      if (l2(ss.d1) <= beta && l2(ss.d2) <= beta) break;
    }
    sigs.push_back(ss);
  }

  AggSig agg;
  agg.d_agg1.assign(m, 0);
  agg.d_agg2.assign(m, 0);
  agg.eta_agg1.assign(n, 0);
  agg.eta_agg2.assign(n, 0);
  for (auto& s : sigs) {
    agg.d_agg1 = vec_add(agg.d_agg1, s.d1, q);
    agg.d_agg2 = vec_add(agg.d_agg2, s.d2, q);
    agg.eta_agg1 = vec_add(agg.eta_agg1, s.eta1, q);
    agg.eta_agg2 = vec_add(agg.eta_agg2, s.eta2, q);
    agg.hs.push_back(s.h);
  }

  std::vector<int64> sum1(n, 0), sum2(n, 0);
  for (size_t i = 0; i < vehs.size(); ++i) {
    sum1 = vec_add(sum1, mat_short_mul(vehs[i].ID1, agg.hs[i], q), q);
    sum2 = vec_add(sum2, mat_short_mul(vehs[i].U, agg.hs[i], q), q);
  }
  std::vector<int64> A1da = mat_vec(A1, agg.d_agg1, q);
  std::vector<int64> Ada = mat_vec(A, agg.d_agg2, q);
  bool a1 = (vec_sub(A1da, sum1, q) == agg.eta_agg1);
  bool a2 = (vec_sub(Ada, sum2, q) == agg.eta_agg2);



  AggSig forge;
  forge.d_agg1.assign(m, 0);
  forge.d_agg2.assign(m, 0);
  std::vector<int64> s1(n, 0), s2(n, 0);
  for (int i = 0; i < vehicles; ++i) {
    std::vector<int64> hp = sample_challenge(k, P.dh, rng);
    forge.hs.push_back(hp);
    s1 = vec_add(s1, mat_short_mul(vehs[i].ID1, hp, q), q);
    s2 = vec_add(s2, mat_short_mul(vehs[i].U, hp, q), q);
  }
  for (auto& x : s1) x = fmodq(-x, q);
  for (auto& x : s2) x = fmodq(-x, q);
  forge.eta_agg1 = s1;
  forge.eta_agg2 = s2;

  std::vector<int64> f1(n, 0), f2(n, 0);
  for (size_t i = 0; i < vehs.size(); ++i) {
    f1 = vec_add(f1, mat_short_mul(vehs[i].ID1, forge.hs[i], q), q);
    f2 = vec_add(f2, mat_short_mul(vehs[i].U, forge.hs[i], q), q);
  }
  bool fa1 = (vec_sub(mat_vec(A1, forge.d_agg1, q), f1, q) == forge.eta_agg1);
  bool fa2 = (vec_sub(mat_vec(A, forge.d_agg2, q), f2, q) == forge.eta_agg2);

  printf("\nExample honest sign (components eta1, eta2, d1, d2, h)\n");
  printf("  eta1[0..2] = {%ld, %ld, %ld}\n", sig.eta1[0], sig.eta1[1], sig.eta1[2]);
  printf("  eta2[0..2] = {%ld, %ld, %ld}\n", sig.eta2[0], sig.eta2[1], sig.eta2[2]);
  printf("  d1[0..2]   = {%ld, %ld, %ld}\n", sig.d1[0], sig.d1[1], sig.d1[2]);
  printf("  d2[0..2]   = {%ld, %ld, %ld}\n", sig.d2[0], sig.d2[1], sig.d2[2]);
  printf("  h[0..2]    = {%ld, %ld, %ld}\n", sig.h[0], sig.h[1], sig.h[2]);
  printf("Honest batch (%d) : accepted\n", vehicles);
  printf("Forged batch    : accepted (checks %d and %d)\n", (int)fa1, (int)fa2);
  printf("Example forged aggregate sign (components d_agg, eta_agg, h')\n");
  printf("  d_agg1[0..2] = {%ld, %ld, %ld}\n",
         forge.d_agg1[0], forge.d_agg1[1], forge.d_agg1[2]);
  printf("  d_agg2[0..2] = {%ld, %ld, %ld}\n",
         forge.d_agg2[0], forge.d_agg2[1], forge.d_agg2[2]);
  printf("  eta_agg1[0..2] = {%ld, %ld, %ld}\n",
         forge.eta_agg1[0], forge.eta_agg1[1], forge.eta_agg1[2]);
  printf("  eta_agg2[0..2] = {%ld, %ld, %ld}\n",
         forge.eta_agg2[0], forge.eta_agg2[1], forge.eta_agg2[2]);
  printf("  h'[0..2]     = {%ld, %ld, %ld} (attacker-chosen)\n",
         forge.hs[0][0], forge.hs[0][1], forge.hs[0][2]);
  printf("\nVerdict: forgery accepted, no secret key used\n");
  return (honest_single && (a1 && a2) && (fa1 && fa2)) ? 0 : 1;
}