// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define CUDA
#else // OpenCL
  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
#endif

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

#ifdef CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.
extern LOCAL uchar cuda_shared[];

typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif


#define FIELD_limb ulong
#define FIELD_LIMBS 6
#define FIELD_LIMB_BITS 64
#define FIELD_INV 9940570264628428797
typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;
CONSTANT FIELD FIELD_ONE = { { 8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861, 6631298214892334189, 1582556514881692819 } };
CONSTANT FIELD FIELD_P = { { 13402431016077863595, 2210141511517208575, 7435674573564081700, 7239337960414712511, 5412103778470702295, 1873798617647539866 } };
CONSTANT FIELD FIELD_R2 = { { 17644856173732828998, 754043588434789617, 10224657059481499349, 7488229067341005760, 11130996698012816685, 1267921511277847466 } };
CONSTANT FIELD FIELD_ZERO = { { 0, 0, 0, 0, 0, 0 } };
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE FIELD FIELD_sub_nvidia(FIELD a, FIELD b) {
asm("sub.cc.u64 %0, %0, %6;\r\n"
"subc.cc.u64 %1, %1, %7;\r\n"
"subc.cc.u64 %2, %2, %8;\r\n"
"subc.cc.u64 %3, %3, %9;\r\n"
"subc.cc.u64 %4, %4, %10;\r\n"
"subc.u64 %5, %5, %11;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
return a;
}
DEVICE FIELD FIELD_add_nvidia(FIELD a, FIELD b) {
asm("add.cc.u64 %0, %0, %6;\r\n"
"addc.cc.u64 %1, %1, %7;\r\n"
"addc.cc.u64 %2, %2, %8;\r\n"
"addc.cc.u64 %3, %3, %9;\r\n"
"addc.cc.u64 %4, %4, %10;\r\n"
"addc.u64 %5, %5, %11;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define FIELD_BITS (FIELD_LIMBS * FIELD_LIMB_BITS)
#if FIELD_LIMB_BITS == 32
  #define FIELD_mac_with_carry mac_with_carry_32
  #define FIELD_add_with_carry add_with_carry_32
#elif FIELD_LIMB_BITS == 64
  #define FIELD_mac_with_carry mac_with_carry_64
  #define FIELD_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool FIELD_gte(FIELD a, FIELD b) {
  for(char i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool FIELD_eq(FIELD a, FIELD b) {
  for(uchar i = 0; i < FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define FIELD_add_ FIELD_add_nvidia
  #define FIELD_sub_ FIELD_sub_nvidia
#else
  DEVICE FIELD FIELD_add_(FIELD a, FIELD b) {
    bool carry = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  FIELD FIELD_sub_(FIELD a, FIELD b) {
    bool borrow = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE FIELD FIELD_sub(FIELD a, FIELD b) {
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, FIELD_P);
  return res;
}

// Modular addition
DEVICE FIELD FIELD_add(FIELD a, FIELD b) {
  FIELD res = FIELD_add_(a, b);
  if(FIELD_gte(res, FIELD_P)) res = FIELD_sub_(res, FIELD_P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void FIELD_reduce(uint32_t accLow[FIELD_LIMBS], uint32_t np0, uint32_t fq[FIELD_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = FIELD_LIMBS;
  uint32_t accHigh[FIELD_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void FIELD_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = FIELD_LIMBS;
  const uint32_t yLimbs  = FIELD_LIMBS;
  const uint32_t xyLimbs = FIELD_LIMBS * 2;
  uint32_t temp[FIELD_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE FIELD FIELD_mul_nvidia(FIELD a, FIELD b) {
  // Perform full multiply
  limb ab[2 * FIELD_LIMBS];
  FIELD_mult_v1(a.val, b.val, ab);

  uint32_t io[FIELD_LIMBS];
  #pragma unroll
  for(int i=0;i<FIELD_LIMBS;i++) {
    io[i]=ab[i];
  }
  FIELD_reduce(io, FIELD_INV, FIELD_P.val);

  // Add io to the upper words of ab
  ab[FIELD_LIMBS] = add_cc(ab[FIELD_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < FIELD_LIMBS - 1; j++) {
    ab[j + FIELD_LIMBS] = addc_cc(ab[j + FIELD_LIMBS], io[j]);
  }
  ab[2 * FIELD_LIMBS - 1] = addc(ab[2 * FIELD_LIMBS - 1], io[FIELD_LIMBS - 1]);

  FIELD r;
  #pragma unroll
  for (int i = 0; i < FIELD_LIMBS; i++) {
    r.val[i] = ab[i + FIELD_LIMBS];
  }

  if (FIELD_gte(r, FIELD_P)) {
    r = FIELD_sub_(r, FIELD_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE FIELD FIELD_mul_default(FIELD a, FIELD b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  FIELD_limb t[FIELD_LIMBS + 2] = {0};
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    FIELD_limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++)
      t[j] = FIELD_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[FIELD_LIMBS] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS + 1] = carry;

    carry = 0;
    FIELD_limb m = FIELD_INV * t[0];
    FIELD_mac_with_carry(m, FIELD_P.val[0], t[0], &carry);
    for(uchar j = 1; j < FIELD_LIMBS; j++)
      t[j - 1] = FIELD_mac_with_carry(m, FIELD_P.val[j], t[j], &carry);

    t[FIELD_LIMBS - 1] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS] = t[FIELD_LIMBS + 1] + carry;
  }

  FIELD result;
  for(uchar i = 0; i < FIELD_LIMBS; i++) result.val[i] = t[i];

  if(FIELD_gte(result, FIELD_P)) result = FIELD_sub_(result, FIELD_P);

  return result;
}

#ifdef CUDA
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_nvidia(a, b);
}
#else
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE FIELD FIELD_sqr(FIELD a) {
  return FIELD_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
DEVICE FIELD FIELD_double(FIELD a) {
  for(uchar i = FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (FIELD_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE FIELD FIELD_pow(FIELD base, uint exponent) {
  FIELD res = FIELD_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, base);
    exponent = exponent >> 1;
    base = FIELD_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE FIELD FIELD_pow_lookup(GLOBAL FIELD *bases, uint exponent) {
  FIELD res = FIELD_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE FIELD FIELD_mont(FIELD a) {
  return FIELD_mul(a, FIELD_R2);
}

DEVICE FIELD FIELD_unmont(FIELD a) {
  FIELD one = FIELD_ZERO;
  one.val[0] = 1;
  return FIELD_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool FIELD_get_bit(FIELD l, uint i) {
  return (l.val[FIELD_LIMBS - 1 - i / FIELD_LIMB_BITS] >> (FIELD_LIMB_BITS - 1 - (i % FIELD_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint FIELD_get_bits(FIELD l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= FIELD_get_bit(l, skip + i);
  }
  return ret;
}
#define EXPONENT_limb ulong
#define EXPONENT_LIMBS 4
#define EXPONENT_LIMB_BITS 64
#define EXPONENT_INV 18446744069414584319
typedef struct { EXPONENT_limb val[EXPONENT_LIMBS]; } EXPONENT;
CONSTANT EXPONENT EXPONENT_ONE = { { 8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911 } };
CONSTANT EXPONENT EXPONENT_P = { { 18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352 } };
CONSTANT EXPONENT EXPONENT_R2 = { { 14526898881837571181, 3129137299524312099, 419701826671360399, 524908885293268753 } };
CONSTANT EXPONENT EXPONENT_ZERO = { { 0, 0, 0, 0 } };
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE EXPONENT EXPONENT_sub_nvidia(EXPONENT a, EXPONENT b) {
asm("sub.cc.u64 %0, %0, %4;\r\n"
"subc.cc.u64 %1, %1, %5;\r\n"
"subc.cc.u64 %2, %2, %6;\r\n"
"subc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
DEVICE EXPONENT EXPONENT_add_nvidia(EXPONENT a, EXPONENT b) {
asm("add.cc.u64 %0, %0, %4;\r\n"
"addc.cc.u64 %1, %1, %5;\r\n"
"addc.cc.u64 %2, %2, %6;\r\n"
"addc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define EXPONENT_BITS (EXPONENT_LIMBS * EXPONENT_LIMB_BITS)
#if EXPONENT_LIMB_BITS == 32
  #define EXPONENT_mac_with_carry mac_with_carry_32
  #define EXPONENT_add_with_carry add_with_carry_32
#elif EXPONENT_LIMB_BITS == 64
  #define EXPONENT_mac_with_carry mac_with_carry_64
  #define EXPONENT_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool EXPONENT_gte(EXPONENT a, EXPONENT b) {
  for(char i = EXPONENT_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool EXPONENT_eq(EXPONENT a, EXPONENT b) {
  for(uchar i = 0; i < EXPONENT_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define EXPONENT_add_ EXPONENT_add_nvidia
  #define EXPONENT_sub_ EXPONENT_sub_nvidia
#else
  DEVICE EXPONENT EXPONENT_add_(EXPONENT a, EXPONENT b) {
    bool carry = 0;
    for(uchar i = 0; i < EXPONENT_LIMBS; i++) {
      EXPONENT_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  EXPONENT EXPONENT_sub_(EXPONENT a, EXPONENT b) {
    bool borrow = 0;
    for(uchar i = 0; i < EXPONENT_LIMBS; i++) {
      EXPONENT_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE EXPONENT EXPONENT_sub(EXPONENT a, EXPONENT b) {
  EXPONENT res = EXPONENT_sub_(a, b);
  if(!EXPONENT_gte(a, b)) res = EXPONENT_add_(res, EXPONENT_P);
  return res;
}

// Modular addition
DEVICE EXPONENT EXPONENT_add(EXPONENT a, EXPONENT b) {
  EXPONENT res = EXPONENT_add_(a, b);
  if(EXPONENT_gte(res, EXPONENT_P)) res = EXPONENT_sub_(res, EXPONENT_P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void EXPONENT_reduce(uint32_t accLow[EXPONENT_LIMBS], uint32_t np0, uint32_t fq[EXPONENT_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = EXPONENT_LIMBS;
  uint32_t accHigh[EXPONENT_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void EXPONENT_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = EXPONENT_LIMBS;
  const uint32_t yLimbs  = EXPONENT_LIMBS;
  const uint32_t xyLimbs = EXPONENT_LIMBS * 2;
  uint32_t temp[EXPONENT_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE EXPONENT EXPONENT_mul_nvidia(EXPONENT a, EXPONENT b) {
  // Perform full multiply
  limb ab[2 * EXPONENT_LIMBS];
  EXPONENT_mult_v1(a.val, b.val, ab);

  uint32_t io[EXPONENT_LIMBS];
  #pragma unroll
  for(int i=0;i<EXPONENT_LIMBS;i++) {
    io[i]=ab[i];
  }
  EXPONENT_reduce(io, EXPONENT_INV, EXPONENT_P.val);

  // Add io to the upper words of ab
  ab[EXPONENT_LIMBS] = add_cc(ab[EXPONENT_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < EXPONENT_LIMBS - 1; j++) {
    ab[j + EXPONENT_LIMBS] = addc_cc(ab[j + EXPONENT_LIMBS], io[j]);
  }
  ab[2 * EXPONENT_LIMBS - 1] = addc(ab[2 * EXPONENT_LIMBS - 1], io[EXPONENT_LIMBS - 1]);

  EXPONENT r;
  #pragma unroll
  for (int i = 0; i < EXPONENT_LIMBS; i++) {
    r.val[i] = ab[i + EXPONENT_LIMBS];
  }

  if (EXPONENT_gte(r, EXPONENT_P)) {
    r = EXPONENT_sub_(r, EXPONENT_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE EXPONENT EXPONENT_mul_default(EXPONENT a, EXPONENT b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  EXPONENT_limb t[EXPONENT_LIMBS + 2] = {0};
  for(uchar i = 0; i < EXPONENT_LIMBS; i++) {
    EXPONENT_limb carry = 0;
    for(uchar j = 0; j < EXPONENT_LIMBS; j++)
      t[j] = EXPONENT_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[EXPONENT_LIMBS] = EXPONENT_add_with_carry(t[EXPONENT_LIMBS], &carry);
    t[EXPONENT_LIMBS + 1] = carry;

    carry = 0;
    EXPONENT_limb m = EXPONENT_INV * t[0];
    EXPONENT_mac_with_carry(m, EXPONENT_P.val[0], t[0], &carry);
    for(uchar j = 1; j < EXPONENT_LIMBS; j++)
      t[j - 1] = EXPONENT_mac_with_carry(m, EXPONENT_P.val[j], t[j], &carry);

    t[EXPONENT_LIMBS - 1] = EXPONENT_add_with_carry(t[EXPONENT_LIMBS], &carry);
    t[EXPONENT_LIMBS] = t[EXPONENT_LIMBS + 1] + carry;
  }

  EXPONENT result;
  for(uchar i = 0; i < EXPONENT_LIMBS; i++) result.val[i] = t[i];

  if(EXPONENT_gte(result, EXPONENT_P)) result = EXPONENT_sub_(result, EXPONENT_P);

  return result;
}

#ifdef CUDA
DEVICE EXPONENT EXPONENT_mul(EXPONENT a, EXPONENT b) {
  return EXPONENT_mul_nvidia(a, b);
}
#else
DEVICE EXPONENT EXPONENT_mul(EXPONENT a, EXPONENT b) {
  return EXPONENT_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE EXPONENT EXPONENT_sqr(EXPONENT a) {
  return EXPONENT_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of EXPONENT_add(a, a)
DEVICE EXPONENT EXPONENT_double(EXPONENT a) {
  for(uchar i = EXPONENT_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (EXPONENT_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(EXPONENT_gte(a, EXPONENT_P)) a = EXPONENT_sub_(a, EXPONENT_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE EXPONENT EXPONENT_pow(EXPONENT base, uint exponent) {
  EXPONENT res = EXPONENT_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = EXPONENT_mul(res, base);
    exponent = exponent >> 1;
    base = EXPONENT_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE EXPONENT EXPONENT_pow_lookup(GLOBAL EXPONENT *bases, uint exponent) {
  EXPONENT res = EXPONENT_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = EXPONENT_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE EXPONENT EXPONENT_mont(EXPONENT a) {
  return EXPONENT_mul(a, EXPONENT_R2);
}

DEVICE EXPONENT EXPONENT_unmont(EXPONENT a) {
  EXPONENT one = EXPONENT_ZERO;
  one.val[0] = 1;
  return EXPONENT_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool EXPONENT_get_bit(EXPONENT l, uint i) {
  return (l.val[EXPONENT_LIMBS - 1 - i / EXPONENT_LIMB_BITS] >> (EXPONENT_LIMB_BITS - 1 - (i % EXPONENT_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint EXPONENT_get_bits(EXPONENT l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= EXPONENT_get_bit(l, skip + i);
  }
  return ret;
}


// Fp2 Extension Field where u^2 + 1 = 0

#define blstrs__fp2__Fp2_LIMB_BITS FIELD_LIMB_BITS
#define blstrs__fp2__Fp2_ZERO ((blstrs__fp2__Fp2){FIELD_ZERO, FIELD_ZERO})
#define blstrs__fp2__Fp2_ONE ((blstrs__fp2__Fp2){FIELD_ONE, FIELD_ZERO})

typedef struct {
  FIELD c0;
  FIELD c1;
} blstrs__fp2__Fp2; // Represents: c0 + u * c1

DEVICE bool blstrs__fp2__Fp2_eq(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  return FIELD_eq(a.c0, b.c0) && FIELD_eq(a.c1, b.c1);
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  a.c0 = FIELD_sub(a.c0, b.c0);
  a.c1 = FIELD_sub(a.c1, b.c1);
  return a;
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_add(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  a.c0 = FIELD_add(a.c0, b.c0);
  a.c1 = FIELD_add(a.c1, b.c1);
  return a;
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_double(blstrs__fp2__Fp2 a) {
  a.c0 = FIELD_double(a.c0);
  a.c1 = FIELD_double(a.c1);
  return a;
}

/*
 * (a_0 + u * a_1)(b_0 + u * b_1) = a_0 * b_0 - a_1 * b_1 + u * (a_0 * b_1 + a_1 * b_0)
 * Therefore:
 * c_0 = a_0 * b_0 - a_1 * b_1
 * c_1 = (a_0 * b_1 + a_1 * b_0) = (a_0 + a_1) * (b_0 + b_1) - a_0 * b_0 - a_1 * b_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  const FIELD aa = FIELD_mul(a.c0, b.c0);
  const FIELD bb = FIELD_mul(a.c1, b.c1);
  const FIELD o = FIELD_add(b.c0, b.c1);
  a.c1 = FIELD_add(a.c1, a.c0);
  a.c1 = FIELD_mul(a.c1, o);
  a.c1 = FIELD_sub(a.c1, aa);
  a.c1 = FIELD_sub(a.c1, bb);
  a.c0 = FIELD_sub(aa, bb);
  return a;
}

/*
 * (a_0 + u * a_1)(a_0 + u * a_1) = a_0 ^ 2 - a_1 ^ 2 + u * 2 * a_0 * a_1
 * Therefore:
 * c_0 = (a_0 * a_0 - a_1 * a_1) = (a_0 + a_1)(a_0 - a_1)
 * c_1 = 2 * a_0 * a_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sqr(blstrs__fp2__Fp2 a) {
  const FIELD ab = FIELD_mul(a.c0, a.c1);
  const FIELD c0c1 = FIELD_add(a.c0, a.c1);
  a.c0 = FIELD_mul(FIELD_sub(a.c0, a.c1), c0c1);
  a.c1 = FIELD_double(ab);
  return a;
}


/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void EXPONENT_radix_fft(GLOBAL EXPONENT* x, // Source buffer
                      GLOBAL EXPONENT* y, // Destination buffer
                      GLOBAL EXPONENT* pq, // Precalculated twiddle factors
                      GLOBAL EXPONENT* omegas, // [omega, omega^2, omega^4, ...]
                      LOCAL EXPONENT* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.
#ifdef CUDA
  // There can only be a single dynamic shared memory item, hence cast it to the type we need.
  EXPONENT* u = (EXPONENT*)cuda_shared;
#else
  LOCAL EXPONENT* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const EXPONENT twiddle = EXPONENT_pow_lookup(omegas, (n >> lgp >> deg) * k);
  EXPONENT tmp = EXPONENT_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = EXPONENT_mul(tmp, x[i*t]);
    tmp = EXPONENT_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = EXPONENT_add(u[i0], u[i1]);
      u[i1] = EXPONENT_sub(tmp, u[i1]);
      if(di != 0) u[i1] = EXPONENT_mul(pq[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
KERNEL void EXPONENT_mul_by_field(GLOBAL EXPONENT* elements,
                        uint n,
                        EXPONENT field) {
  const uint gid = GET_GLOBAL_ID();
  elements[gid] = EXPONENT_mul(elements[gid], field);
}


// Elliptic curve operations (Short Weierstrass Jacobian form)

#define POINT_ZERO ((POINT_jacobian){FIELD_ZERO, FIELD_ONE, FIELD_ZERO})

typedef struct {
  FIELD x;
  FIELD y;
} POINT_affine;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD z;
} POINT_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE POINT_jacobian POINT_double(POINT_jacobian inp) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(inp.z, local_zero)) {
      return inp;
  }

  const FIELD a = FIELD_sqr(inp.x); // A = X1^2
  const FIELD b = FIELD_sqr(inp.y); // B = Y1^2
  FIELD c = FIELD_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  FIELD d = FIELD_add(inp.x, b);
  d = FIELD_sqr(d); d = FIELD_sub(FIELD_sub(d, a), c); d = FIELD_double(d);

  const FIELD e = FIELD_add(FIELD_double(a), a); // E = 3*A
  const FIELD f = FIELD_sqr(e);

  inp.z = FIELD_mul(inp.y, inp.z); inp.z = FIELD_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = FIELD_sub(FIELD_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = FIELD_double(c); c = FIELD_double(c); c = FIELD_double(c);
  inp.y = FIELD_sub(FIELD_mul(FIELD_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE POINT_jacobian POINT_add_mixed(POINT_jacobian a, POINT_affine b) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) {
    const FIELD local_one = FIELD_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const FIELD z1z1 = FIELD_sqr(a.z);
  const FIELD u2 = FIELD_mul(b.x, z1z1);
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1);

  if(FIELD_eq(a.x, u2) && FIELD_eq(a.y, s2)) {
      return POINT_double(a);
  }

  const FIELD h = FIELD_sub(u2, a.x); // H = U2-X1
  const FIELD hh = FIELD_sqr(h); // HH = H^2
  FIELD i = FIELD_double(hh); i = FIELD_double(i); // I = 4*HH
  FIELD j = FIELD_mul(h, i); // J = H*I
  FIELD r = FIELD_sub(s2, a.y); r = FIELD_double(r); // r = 2*(S2-Y1)
  const FIELD v = FIELD_mul(a.x, i);

  POINT_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = FIELD_sub(FIELD_sub(FIELD_sqr(r), j), FIELD_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = FIELD_mul(a.y, j); j = FIELD_double(j);
  ret.y = FIELD_sub(FIELD_mul(FIELD_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = FIELD_add(a.z, h); ret.z = FIELD_sub(FIELD_sub(FIELD_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE POINT_jacobian POINT_add(POINT_jacobian a, POINT_jacobian b) {

  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) return b;
  if(FIELD_eq(b.z, local_zero)) return a;

  const FIELD z1z1 = FIELD_sqr(a.z); // Z1Z1 = Z1^2
  const FIELD z2z2 = FIELD_sqr(b.z); // Z2Z2 = Z2^2
  const FIELD u1 = FIELD_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const FIELD u2 = FIELD_mul(b.x, z1z1); // U2 = X2*Z1Z1
  FIELD s1 = FIELD_mul(FIELD_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(FIELD_eq(u1, u2) && FIELD_eq(s1, s2))
    return POINT_double(a);
  else {
    const FIELD h = FIELD_sub(u2, u1); // H = U2-U1
    FIELD i = FIELD_double(h); i = FIELD_sqr(i); // I = (2*H)^2
    const FIELD j = FIELD_mul(h, i); // J = H*I
    FIELD r = FIELD_sub(s2, s1); r = FIELD_double(r); // r = 2*(S2-S1)
    const FIELD v = FIELD_mul(u1, i); // V = U1*I
    a.x = FIELD_sub(FIELD_sub(FIELD_sub(FIELD_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = FIELD_mul(FIELD_sub(v, a.x), r);
    s1 = FIELD_mul(s1, j); s1 = FIELD_double(s1); // S1 = S1 * J * 2
    a.y = FIELD_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = FIELD_add(a.z, b.z); a.z = FIELD_sqr(a.z);
    a.z = FIELD_sub(FIELD_sub(a.z, z1z1), z2z2);
    a.z = FIELD_mul(a.z, h);

    return a;
  }
}
/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void POINT_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *buckets,
    GLOBAL POINT_jacobian *results,
    GLOBAL EXPONENT *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const POINT_jacobian local_zero = POINT_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  const uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_jacobian res = POINT_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = EXPONENT_get_bits(exps[i], bits, w);

    #if defined(OPENCL_NVIDIA) || defined(CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = POINT_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  POINT_jacobian acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}
// Elliptic curve operations (Short Weierstrass Jacobian form)

#define blstrs__g2__G2Affine_ZERO ((blstrs__g2__G2Affine_jacobian){blstrs__fp2__Fp2_ZERO, blstrs__fp2__Fp2_ONE, blstrs__fp2__Fp2_ZERO})

typedef struct {
  blstrs__fp2__Fp2 x;
  blstrs__fp2__Fp2 y;
} blstrs__g2__G2Affine_affine;

typedef struct {
  blstrs__fp2__Fp2 x;
  blstrs__fp2__Fp2 y;
  blstrs__fp2__Fp2 z;
} blstrs__g2__G2Affine_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_double(blstrs__g2__G2Affine_jacobian inp) {
  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(inp.z, local_zero)) {
      return inp;
  }

  const blstrs__fp2__Fp2 a = blstrs__fp2__Fp2_sqr(inp.x); // A = X1^2
  const blstrs__fp2__Fp2 b = blstrs__fp2__Fp2_sqr(inp.y); // B = Y1^2
  blstrs__fp2__Fp2 c = blstrs__fp2__Fp2_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  blstrs__fp2__Fp2 d = blstrs__fp2__Fp2_add(inp.x, b);
  d = blstrs__fp2__Fp2_sqr(d); d = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(d, a), c); d = blstrs__fp2__Fp2_double(d);

  const blstrs__fp2__Fp2 e = blstrs__fp2__Fp2_add(blstrs__fp2__Fp2_double(a), a); // E = 3*A
  const blstrs__fp2__Fp2 f = blstrs__fp2__Fp2_sqr(e);

  inp.z = blstrs__fp2__Fp2_mul(inp.y, inp.z); inp.z = blstrs__fp2__Fp2_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = blstrs__fp2__Fp2_double(c); c = blstrs__fp2__Fp2_double(c); c = blstrs__fp2__Fp2_double(c);
  inp.y = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add_mixed(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_affine b) {
  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(a.z, local_zero)) {
    const blstrs__fp2__Fp2 local_one = blstrs__fp2__Fp2_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const blstrs__fp2__Fp2 z1z1 = blstrs__fp2__Fp2_sqr(a.z);
  const blstrs__fp2__Fp2 u2 = blstrs__fp2__Fp2_mul(b.x, z1z1);
  const blstrs__fp2__Fp2 s2 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(b.y, a.z), z1z1);

  if(blstrs__fp2__Fp2_eq(a.x, u2) && blstrs__fp2__Fp2_eq(a.y, s2)) {
      return blstrs__g2__G2Affine_double(a);
  }

  const blstrs__fp2__Fp2 h = blstrs__fp2__Fp2_sub(u2, a.x); // H = U2-X1
  const blstrs__fp2__Fp2 hh = blstrs__fp2__Fp2_sqr(h); // HH = H^2
  blstrs__fp2__Fp2 i = blstrs__fp2__Fp2_double(hh); i = blstrs__fp2__Fp2_double(i); // I = 4*HH
  blstrs__fp2__Fp2 j = blstrs__fp2__Fp2_mul(h, i); // J = H*I
  blstrs__fp2__Fp2 r = blstrs__fp2__Fp2_sub(s2, a.y); r = blstrs__fp2__Fp2_double(r); // r = 2*(S2-Y1)
  const blstrs__fp2__Fp2 v = blstrs__fp2__Fp2_mul(a.x, i);

  blstrs__g2__G2Affine_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(r), j), blstrs__fp2__Fp2_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = blstrs__fp2__Fp2_mul(a.y, j); j = blstrs__fp2__Fp2_double(j);
  ret.y = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = blstrs__fp2__Fp2_add(a.z, h); ret.z = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_jacobian b) {

  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(a.z, local_zero)) return b;
  if(blstrs__fp2__Fp2_eq(b.z, local_zero)) return a;

  const blstrs__fp2__Fp2 z1z1 = blstrs__fp2__Fp2_sqr(a.z); // Z1Z1 = Z1^2
  const blstrs__fp2__Fp2 z2z2 = blstrs__fp2__Fp2_sqr(b.z); // Z2Z2 = Z2^2
  const blstrs__fp2__Fp2 u1 = blstrs__fp2__Fp2_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const blstrs__fp2__Fp2 u2 = blstrs__fp2__Fp2_mul(b.x, z1z1); // U2 = X2*Z1Z1
  blstrs__fp2__Fp2 s1 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const blstrs__fp2__Fp2 s2 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(blstrs__fp2__Fp2_eq(u1, u2) && blstrs__fp2__Fp2_eq(s1, s2))
    return blstrs__g2__G2Affine_double(a);
  else {
    const blstrs__fp2__Fp2 h = blstrs__fp2__Fp2_sub(u2, u1); // H = U2-U1
    blstrs__fp2__Fp2 i = blstrs__fp2__Fp2_double(h); i = blstrs__fp2__Fp2_sqr(i); // I = (2*H)^2
    const blstrs__fp2__Fp2 j = blstrs__fp2__Fp2_mul(h, i); // J = H*I
    blstrs__fp2__Fp2 r = blstrs__fp2__Fp2_sub(s2, s1); r = blstrs__fp2__Fp2_double(r); // r = 2*(S2-S1)
    const blstrs__fp2__Fp2 v = blstrs__fp2__Fp2_mul(u1, i); // V = U1*I
    a.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(v, a.x), r);
    s1 = blstrs__fp2__Fp2_mul(s1, j); s1 = blstrs__fp2__Fp2_double(s1); // S1 = S1 * J * 2
    a.y = blstrs__fp2__Fp2_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = blstrs__fp2__Fp2_add(a.z, b.z); a.z = blstrs__fp2__Fp2_sqr(a.z);
    a.z = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(a.z, z1z1), z2z2);
    a.z = blstrs__fp2__Fp2_mul(a.z, h);

    return a;
  }
}
/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void blstrs__g2__G2Affine_multiexp(
    GLOBAL blstrs__g2__G2Affine_affine *bases,
    GLOBAL blstrs__g2__G2Affine_jacobian *buckets,
    GLOBAL blstrs__g2__G2Affine_jacobian *results,
    GLOBAL EXPONENT *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const blstrs__g2__G2Affine_jacobian local_zero = blstrs__g2__G2Affine_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  const uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  blstrs__g2__G2Affine_jacobian res = blstrs__g2__G2Affine_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = EXPONENT_get_bits(exps[i], bits, w);

    #if defined(OPENCL_NVIDIA) || defined(CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = blstrs__g2__G2Affine_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = blstrs__g2__G2Affine_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = blstrs__g2__G2Affine_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  blstrs__g2__G2Affine_jacobian acc = blstrs__g2__G2Affine_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = blstrs__g2__G2Affine_add(acc, buckets[j]);
    res = blstrs__g2__G2Affine_add(res, acc);
  }

  results[gid] = res;
}



