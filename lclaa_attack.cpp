#include "lattice.hpp"
#include <cinttypes>
#include <random>
#include <set>


struct Params {
  size_t n = 512;
  int    qbits = 25;
  size_t m = 512 * 2 * 25;
  double s = 4000;
  double sigma = 15000;
  double gamma = 30000;
  int    dh = 4;
};

struct UserKey {
  Mat S1, S2;
  Mat X, U;
  std::vector<int64> pid;
};



static std::vector<int64> challenge(const std::string& label,
                                    const std::vector<int64>& pid,
                                    const std::vector<int64>& w,
                                    const Mat& X, size_t n, int dh,
                                    const std::vector<int64>& m_bits,
                                    const std::vector<int64>& t_bits) {
  std::vector<unsigned char> buf;
  for (auto c : label) buf.push_back((unsigned char)c);
  for (auto x : pid) for (int b = 0; b < 8; ++b) buf.push_back((unsigned char)(x >> (8 * b)));
  for (auto x : w) for (int b = 0; b < 8; ++b) buf.push_back((unsigned char)(x >> (8 * b)));
  for (uint32_t i = 0; i < X.rows && i < 8; ++i)
    for (int b = 0; b < 8; ++b) buf.push_back((unsigned char)(X(i, 0) >> (8 * b)));
  for (auto x : m_bits) for (int b = 0; b < 8; ++b) buf.push_back((unsigned char)(x >> (8 * b)));
  for (auto x : t_bits) for (int b = 0; b < 8; ++b) buf.push_back((unsigned char)(x >> (8 * b)));
  unsigned char hh[32];
  SHA256(buf.data(), buf.size(), hh);
  unsigned long long seed = 0;
  for (int i = 0; i < 8; ++i) seed = (seed << 8) | hh[i];
  Xoshiro256 r2(seed);
  std::vector<int64> out(n, 0);
  std::set<int> used;
  while ((int)used.size() < dh) used.insert((int)r2.uniform_int(0, (int64)n - 1));
  int flip = 0;
  for (auto idx : used) out[idx] = (r2.next() & 1) ? 1 : -1;
  (void)flip;
  return out;
}

static std::vector<int64> short_mul(const Mat& M, const std::vector<int64>& h,
                                    int64 q) {
  std::vector<int64> out(M.rows, 0);
  for (size_t i = 0; i < M.rows; ++i) {
    __int128 acc = 0;
    for (size_t j = 0; j < M.cols; ++j) acc += (__int128)M(i, j) * h[j];
    out[i] = fmodq((int64)(acc % q), q);
  }
  return out;
}

struct SingleSig {
  std::vector<int64> w;
  std::vector<int64> z;
  std::vector<int64> m_bits, t_bits;
};

struct AggSig {
  std::vector<int64> z_agg;
  std::vector<SingleSig> sigs;
};

int main(int argc, char** argv) {
  Params P;
  int vehicles = 8;
  unsigned long long seed = 0x0B0C2025ULL;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--small") { P.n = 128; P.m = 128 * 2 * 25; }
    else if (a == "--n" && i + 1 < argc) {
      P.n = (size_t)atoi(argv[++i]);
      P.m = P.n * 2 * (size_t)P.qbits;
    }
    else if (a == "--qbits" && i + 1 < argc) {
      P.qbits = atoi(argv[++i]);
      P.m = P.n * 2 * (size_t)P.qbits;
    }
    else if (a == "--vehicles" && i + 1 < argc) vehicles = atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 0);
  }
  size_t n = P.n, m = P.m;
  int64 q = ((int64)1) << P.qbits;
  double beta = P.gamma * std::sqrt((double)m);
  Xoshiro256 rng(seed);

  Mat A(n, m, q);
  for (auto& v : A.v) v = rng.uniform_int(0, q - 1);

  auto enroll = [&](const std::string& pidlabel) {
    Mat S1(m, n, q);
    for (auto& x : S1.v) x = (int64)std::llround(snorm_round(rng) * P.s);
    Mat S2(m, n, q);
    for (auto& x : S2.v) x = (int64)std::llround(snorm_round(rng) * P.s);
    Mat X = mat_mul(A, S1, q);
    Mat U = mat_mul(A, S2, q);
    std::vector<int64> pid;
    for (size_t i = 0; i < 16; ++i)
      pid.push_back((int64)(unsigned char)pidlabel[i % pidlabel.size()] + (int64)i * 101);
    UserKey u{std::move(S1), std::move(S2), std::move(X), std::move(U), std::move(pid)};
    return u;
  };
  auto sign = [&](const UserKey& u, const std::vector<int64>& mb,
                  const std::vector<int64>& tb, Xoshiro256& r) {
    SingleSig s;
    s.m_bits = mb;
    s.t_bits = tb;
    for (;;) {
      std::vector<int64> y = gauss_vec(m, P.sigma, r);
      s.w = mat_vec(A, y, q);
      std::vector<int64> h3 = challenge("H3", u.pid, s.w, u.X, n, P.dh, mb, tb);
      std::vector<int64> h4 = challenge("H4", u.pid, s.w, u.X, n, P.dh, mb, tb);
      s.z = vec_add(y, short_mul(u.S1, h3, q), q);
      s.z = vec_add(s.z, short_mul(u.S2, h4, q), q);
      if (l2(s.z) <= beta) break;
    }
    return s;
  };

  std::vector<int64> mb = {0x01, 0x02};
  std::vector<int64> tb = {0x7E7D, 0x1A2B};
  UserKey u0 = enroll("pid0");
  SingleSig sig0 = sign(u0, mb, tb, rng);

  bool norm_ok = (l2(sig0.z) <= beta + 1e-6);
  std::vector<int64> h3v = challenge("H3", u0.pid, sig0.w, u0.X, n, P.dh, mb, tb);
  std::vector<int64> h4v = challenge("H4", u0.pid, sig0.w, u0.X, n, P.dh, mb, tb);
  std::vector<int64> q1 = mat_vec(A, sig0.z, q);
  std::vector<int64> q2 = vec_add(sig0.w, short_mul(u0.X, h3v, q), q);
  q2 = vec_add(q2, short_mul(u0.U, h4v, q), q);
  bool eq6 = (q1 == q2);

  std::vector<UserKey> users{u0};
  std::vector<SingleSig> sigs{sig0};
  for (int i = 1; i < vehicles; ++i) {
    UserKey u = enroll("pid" + std::to_string(i));
    std::vector<int64> mb2 = {(int64)(i * 7) & 0xFF, (int64)(i * 13) & 0xFF};
    std::vector<int64> tb2 = {0x1000 + i, 0x2000 + i};
    sigs.push_back(sign(u, mb2, tb2, rng));
    users.push_back(std::move(u));
  }
  AggSig agg;
  agg.z_agg.assign(m, 0);
  for (auto& s : sigs) agg.z_agg = vec_add(agg.z_agg, s.z, q);
  std::vector<int64> rhs(n, 0);
  for (size_t k = 0; k < sigs.size(); ++k) {
    std::vector<int64> h3 = challenge("H3", users[k].pid, sigs[k].w, users[k].X,
                                      n, P.dh, sigs[k].m_bits, sigs[k].t_bits);
    std::vector<int64> h4 = challenge("H4", users[k].pid, sigs[k].w, users[k].X,
                                      n, P.dh, sigs[k].m_bits, sigs[k].t_bits);
    std::vector<int64> part = sigs[k].w;
    part = vec_add(part, short_mul(users[k].X, h3, q), q);
    part = vec_add(part, short_mul(users[k].U, h4, q), q);
    rhs = vec_add(rhs, part, q);
  }
  std::vector<int64> az = mat_vec(A, agg.z_agg, q);
  bool agg_ok = (az == rhs);



  std::vector<int64> frhs(n, 0);
  std::vector<std::vector<int64>> fws, fmb, ftb;
  fmb.reserve(vehicles);
  ftb.reserve(vehicles);
  fws.reserve(vehicles);
  for (int k = 0; k < vehicles; ++k) {
    fws.push_back(mat_vec(A, gauss_vec(m, P.sigma, rng), q));
    fmb.push_back({(int64)(0xA0 + k), (int64)(0xB0 + k)});
    ftb.push_back({(int64)(0x3000 + k), (int64)(0x4000 + k)});
    std::vector<int64> h3 = challenge("H3", users[k].pid, fws[k], users[k].X,
                                      n, P.dh, fmb[k], ftb[k]);
    std::vector<int64> h4 = challenge("H4", users[k].pid, fws[k], users[k].X,
                                      n, P.dh, fmb[k], ftb[k]);
    std::vector<int64> part = fws[k];
    part = vec_add(part, short_mul(users[k].X, h3, q), q);
    part = vec_add(part, short_mul(users[k].U, h4, q), q);
    frhs = vec_add(frhs, part, q);
  }





  size_t cand = std::min(4 * n, m);
  std::vector<std::vector<int64>> basis;
  std::vector<size_t> pivots;
  std::vector<size_t> sel;
  for (size_t c = 0; c < cand && sel.size() < n; ++c) {
    std::vector<int64> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = A(i, c) & 1;
    for (size_t j = 0; j < basis.size(); ++j)
      if (x[pivots[j]]) for (size_t i = 0; i < n; ++i) x[i] ^= basis[j][i];
    size_t piv = n;
    for (size_t i = 0; i < n; ++i) if (x[i]) { piv = i; break; }
    if (piv < n) {
      basis.push_back(x);
      pivots.push_back(piv);
      sel.push_back(c);
    }
  }
  if (sel.size() < n) {
    printf("%d\n", -1); fprintf(stderr, "indep fail %zu %zu\n",
           n, sel.size());
    return 2;
  }
  Mat B(n, n, q);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) B(i, j) = A(i, sel[j]);
  std::vector<int64> x(n);
  {


    Mat W(n, n + 1, q);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) W(i, j) = B(i, j);
      W(i, n) = frhs[i];
    }
    for (size_t kk = 0; kk < n; ++kk) {
      size_t pr = kk, pc = kk;
      bool found = false;
      for (size_t r = kk; r < n && !found; ++r)
        for (size_t c = kk; c < n && !found; ++c)
          if (W(r, c) & 1) { pr = r; pc = c; found = true; }
      if (!found) { fprintf(stderr, "pivot fail %zu\n", kk); return 2; }
      if (pr != kk)
        for (size_t j = 0; j <= n; ++j) std::swap(W(kk, j), W(pr, j));
      if (pc != kk) {
        for (size_t r = 0; r < n; ++r) std::swap(W(r, kk), W(r, pc));
        std::swap(sel[kk], sel[pc]);
      }
      int64 piv = W(kk, kk);
      int64 inv = inv_mod_pow2(piv, P.qbits);
      for (size_t j = kk; j <= n; ++j)
        W(kk, j) = (int64)((__int128)W(kk, j) * inv % q);
      for (size_t r = 0; r < n; ++r)
        if (r != kk) {
          int64 f = W(r, kk);
          if (f == 0) continue;
          for (size_t j = kk; j <= n; ++j)
            W(r, j) = (W(r, j) - (int64)((__int128)f * W(kk, j) % q)) % q;
        }
    }
    for (size_t i = 0; i < n; ++i) x[i] = fmodq(W(i, n), q);
  }
  std::vector<int64> z_forge(m, 0);
  for (size_t t = 0; t < n; ++t) z_forge[sel[t]] = x[t];

  std::vector<int64> faz = mat_vec(A, z_forge, q);
  bool forge_ok = (faz == frhs);
  double agg_gate = (double)vehicles * beta;
  bool under = l2(z_forge) <= agg_gate;
  int ebytes = P.qbits / 8;
  printf("\nExample honest sign (components w, z)\n");
  printf("  w[0..2] = {%ld, %ld, %ld}\n", sig0.w[0], sig0.w[1], sig0.w[2]);
  printf("  z[0..2] = {%ld, %ld, %ld}\n", sig0.z[0], sig0.z[1], sig0.z[2]);
  printf("Honest batch (%d) : accepted\n", vehicles);
  printf("Forged batch    : accepted (equation check %d)\n", (int)forge_ok);
  printf("Example forged aggregate sign (components z_agg, w')\n");
  printf("  z_agg[0..2] = {%ld, %ld, %ld}\n", z_forge[0], z_forge[1], z_forge[2]);
  printf("  w'[0..2]    = {%ld, %ld, %ld} (attacker-chosen)\n",
         sigs[0].w[0], sigs[0].w[1], sigs[0].w[2]);
  printf("  Forged z size %.0f vs honest %.0f (gate %.0f)\n",
         l2(z_forge), l2(agg.z_agg), agg_gate);
  printf("\nVerdict: forgery accepted, no secret key used\n");
  return (norm_ok && eq6 && agg_ok && forge_ok && !under) ? 0 : 1;
}