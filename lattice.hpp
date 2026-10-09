#ifndef LBCLAA_LATTICE_HPP
#define LBCLAA_LATTICE_HPP

#include <openssl/sha.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Real-scheme support code for the executable forgery plays.
//   * integers `a mod q` are kept in [-q/2, q/2)   (symmetric residues)
//   * matrices are row-major, all products accumulated in __int128 and
//     reduced once (m ~ 6k, q ~ 2^25, so partial sums never overflow)
//   * hashing uses SHA-256 (OpenSSL), matching the surrounding repo
//   * rejection sampling follows Lyubashevsky-style masking

using int64 = std::int64_t;

inline int64 fmodq(int64 a, int64 q) {
  int64 r = a % q;
  if (r < -q / 2) r += q;
  if (r > q / 2) r -= q;
  if (r < 0 && r == -q / 2) r += q;  // keep q/2 as the representative for q/2
  return r;
}

struct Xoshiro256 {
  unsigned long long s[4];
  explicit Xoshiro256(unsigned long long seed) {
    for (int i = 0; i < 4; ++i) {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      s[i] = seed;
    }
  }
  unsigned long long next() {
    unsigned long long res = (s[0] + s[3]) * 5;
    res = ((res << 7) | (res >> 57)) * 9;
    unsigned long long t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> 19);
    return res;
  }
  double uniform() {
    return (next() >> 11) * 0x1.0p-53;
  }
  int64 uniform_int(int64 lo, int64 hi) {  // inclusive
    return lo + (int64)(uniform() * (double)(hi - lo + 1));
  }
};

// Round a standard-normal draw to a discrete integer; z ~ round(N(0,1)).
inline int64 snorm_round(Xoshiro256& rng) {
  double u1 = rng.uniform(), u2 = rng.uniform();
  if (u1 <= 0.0) u1 = 1e-12;
  double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  return (int64)std::llround(z);
}

struct Mat {
  size_t rows, cols;
  int64 q;
  std::vector<int64> v;
  Mat(size_t r = 0, size_t c = 0, int64 mq = 0) : rows(r), cols(c), q(mq), v(r * c, 0) {}
  int64& operator()(size_t i, size_t j) { return v[i * cols + j]; }
  const int64& operator()(size_t i, size_t j) const { return v[i * cols + j]; }
  size_t sz() const { return rows * cols; }
};

// C = A * B  (mod q)  using __int128 accumulation.
// Loop order i,k,j gives contiguous reads of both A and B.  For each output
// row we accumulate into a __int128 scratch row (A.cols products each bounded
// by q^2/4; q < 2^26 and A.cols < ~2^31 keep every partial sum inside
// __int128), then reduce once per cell.
inline Mat mat_mul(const Mat& A, const Mat& B, int64 q) {
  Mat C(A.rows, B.cols, q);
  std::vector<__int128> acc(B.cols, 0);
  for (size_t i = 0; i < A.rows; ++i) {
    std::fill(acc.begin(), acc.end(), 0);
    size_t arow = i * A.cols;
    for (size_t k = 0; k < A.cols; ++k) {
      int64 aik = A.v[arow + k];
      if (aik == 0) continue;
      size_t brow = k * B.cols;
      for (size_t j = 0; j < B.cols; ++j)
        acc[j] += (__int128)aik * B.v[brow + j];
    }
    size_t crow = i * B.cols;
    for (size_t j = 0; j < B.cols; ++j)
      C.v[crow + j] = fmodq((int64)(acc[j] % q), q);
  }
  return C;
}

// y = A * x  (mod q); x has length A.cols.
inline std::vector<int64> mat_vec(const Mat& A, const std::vector<int64>& x,
                                  int64 q) {
  std::vector<int64> y(A.rows, 0);
  for (size_t i = 0; i < A.rows; ++i) {
    __int128 acc = 0;
    for (size_t k = 0; k < A.cols; ++k) acc += (__int128)A(i, k) * x[k];
    y[i] = fmodq((int64)(acc % q), q);
  }
  return y;
}

inline std::vector<int64> mat_vec_t(const Mat& A, const std::vector<int64>& x,
                                    int64 q) {
  std::vector<int64> y(A.cols, 0);
  for (size_t i = 0; i < A.rows; ++i)
    if (x[i] != 0)
      for (size_t j = 0; j < A.cols; ++j)
        y[j] = (y[j] + (__int128)A(i, j) * x[i]) % q;
  for (auto& v : y) v = fmodq(v, q);
  return y;
}

inline std::vector<int64> vec_add(const std::vector<int64>& a,
                                  const std::vector<int64>& b, int64 q) {
  std::vector<int64> c(a.size());
  for (size_t i = 0; i < a.size(); ++i) c[i] = fmodq(a[i] + b[i], q);
  return c;
}
inline std::vector<int64> vec_sub(const std::vector<int64>& a,
                                  const std::vector<int64>& b, int64 q) {
  std::vector<int64> c(a.size());
  for (size_t i = 0; i < a.size(); ++i) c[i] = fmodq(a[i] - b[i], q);
  return c;
}
inline std::vector<int64> vec_mul(const std::vector<int64>& a,
                                  const std::vector<int64>& b, int64 q) {
  std::vector<int64> c(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    c[i] = fmodq((__int128)a[i] * b[i] % q, q);
  return c;
}

// ||v||_2 over integer (signed) entries.
inline double l2(const std::vector<int64>& v) {
  __int128 acc = 0;
  for (auto x : v) acc += (__int128)x * x;
  return std::sqrt((long double)acc);
}

// Modular inverse mod 2^k for odd a, via Newton iteration (Hensel lifting).
inline int64 inv_mod_pow2(int64 a, int k) {
  a = a % (((int64)1) << k);
  if (a < 0) a += ((int64)1) << k;
  int64 b = 1;  // b == a^{-1} mod 2 since a is odd
  int bits = 1;
  while (bits < k) {
    int next = std::min(bits * 2, k);
    int64 mod = ((int64)1) << next;
    b = (b * (2 - (a % mod) * b % mod + mod)) % mod;
    if (b < 0) b += mod;
    bits = next;
  }
  return b & ((((int64)1) << k) - 1);
}

unsigned char sha256_buf(const std::vector<unsigned char>& in,
                         unsigned char out[32]) {
  SHA256(in.data(), in.size(), out);
  return 0;
}

inline std::vector<int64> hash_to_q(const std::string& tag,
                                    const std::vector<std::vector<int64>>& data,
                                    size_t n, int64 q, Xoshiro256* rng = nullptr,
                                    bool randomized = true) {
  unsigned char h[32];
  std::vector<unsigned char> buf;
  for (auto c : tag) buf.push_back((unsigned char)c);
  for (auto& d : data)
    for (auto x : d) {
      unsigned char tmp[8];
      int64 s = x;
      for (int i = 0; i < 8; ++i) { tmp[i] = (unsigned char)(s & 0xFF); s >>= 8; }
      buf.insert(buf.end(), tmp, tmp + 8);
    }
  SHA256(buf.data(), buf.size(), h);
  int64 bits = 0;
  while (((int64)1 << bits) < q) bits++;
  bits = std::min(bits + 8, (int64)62);
  int64 mask = (bits >= 63) ? -1 : (((int64)1 << bits) - 1);
  std::vector<int64> out(n);
  size_t pos = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned long long acc = 0;
    acc |= (unsigned long long)h[pos % 32];
    acc <<= 8;
    acc |= (unsigned long long)h[(pos + 1) % 32];
    pos = (pos + 2) % 32;
    int64 val = (int64)(acc & mask) % q;
    if (val < 0) val += q;
    if (val > q / 2) val -= q;
    if (randomized && rng) val = (val + rng->uniform_int(0, q - 1)) % q;
    out[i] = fmodq(val, q);
  }
  return out;
}

inline std::vector<int64> gauss_vec(size_t m, double sigma, Xoshiro256& rng) {
  std::vector<int64> y(m);
  for (auto& v : y) v = (int64)std::llround(snorm_round(rng) * sigma);
  return y;
}

inline bool reject_test(const std::vector<int64>& delta,
                        const std::vector<int64>& sc, double sigma,
                        double M, Xoshiro256& rng) {
  // accept w.p. min( exp( (||c||^2 - 2<c,delta-c>) / (2 sigma^2) ), 1 )  with
  // the standard M-shaped bound: |z| -> ratio D_sigma(z)/(M*D_{c,sigma}(z)).
  long double num = 0.0, den = 0.0;
  for (size_t i = 0; i < delta.size(); ++i) {
    num += (long double)delta[i] * delta[i];
    long double d = (long double)delta[i] - sc[i];
    den += d * d;
  }
  if (den == 0.0) return true;
  long double ratio = std::exp((num - den) / (2.0 * sigma * sigma));
  if (ratio >= M) return true;
  return rng.uniform() * M < ratio;
}

#endif