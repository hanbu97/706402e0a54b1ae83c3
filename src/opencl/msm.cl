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

/////////////////////////////////////////////////////
// define field
// typedef uint uint32_t;

#define FIELD_limb ulong
#define FIELD_LIMBS 6
#define FIELD_LIMB_BITS 64
typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;
typedef struct { FIELD_limb val[4]; } blst_scalar;

// typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;

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

#define FIELD_INV 0x8508bfffffffffff
CONSTANT FIELD FIELD_P = {{
  0x8508c00000000001, 0x170b5d4430000000,
  0x1ef3622fba094800, 0x1a22d9f300f5138f,
  0xc63b05c06ca1493b, 0x1ae3a4617c510ea
  }};

CONSTANT FIELD FIELD_R2 = {{
  0xb786686c9400cd22, 0x329fcaab00431b1,
  0x22a5f11162d6b46d, 0xbfdf7d03827dc3ac,
  0x837e92f041790bf9, 0x6dfccb1e914b88
  }};

CONSTANT FIELD FIELD_ONE = {{
  0x02cdffffffffff68, 0x51409f837fffffb1,
  0x9f7db3a98a7d3ff2, 0x7b4e97b76e7c6305,
  0x4cf495bf803c84e8, 0x008d6661e2fdf49a
  }};

// #define FIELD_INV 9586122913090633727
// CONSTANT FIELD FIELD_P = {{
//   9586122913090633729, 1660523435060625408,
//   2230234197602682880, 1883307231910630287,
//   14284016967150029115, 121098312706494698
//   }};

// CONSTANT FIELD FIELD_R2 = {{
//   13224372171368877346, 227991066186625457,
//   2496666625421784173, 13825906835078366124,
//   9475172226622360569, 30958721782860680
//   }};

// CONSTANT FIELD FIELD_ONE = {{
//   202099033278250856, 5854854902718660529,
//   11492539364873682930, 8885205928937022213,
//   5545221690922665192, 39800542322357402
//   }};
  
CONSTANT FIELD FIELD_ZERO = { { 0, 0, 0, 0, 0, 0 } };
CONSTANT uint WINDOW_SIZE = 128;

/////////////////////////////////////////////////////
DEVICE int is_blst_p1_zero(const POINT_jacobian p) {
    return p.z.val[0] == 0 &&
        p.z.val[1] == 0 &&
        p.z.val[2] == 0 &&
        p.z.val[3] == 0 &&
        p.z.val[4] == 0 &&
        p.z.val[5] == 0;
}

DEVICE int is_blst_p1_affine_zero(const POINT_affine p) {
    return p.x.val[0] == 0 &&
        p.x.val[1] == 0 &&
        p.x.val[2] == 0 &&
        p.x.val[3] == 0 &&
        p.x.val[4] == 0 &&
        p.x.val[5] == 0;
}

DEVICE int is_blst_fp_eq(const FIELD p1, const FIELD p2) {
    return p1.val[0] == p2.val[0] &&
        p1.val[1] == p2.val[1] &&
        p1.val[2] == p2.val[2] &&
        p1.val[3] == p2.val[3] &&
        p1.val[4] == p2.val[4] &&
        p1.val[5] == p2.val[5];
}

/////////////////////////////////////////////////////
// FIELD_LIMB_BITS == 64

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  ulong lo = a + *b;
  *b = lo < a;
  return lo;
}

#define FIELD_mac_with_carry mac_with_carry_64
#define FIELD_add_with_carry add_with_carry_64

/////////////////////////////////////////////////////

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

DEVICE FIELD FIELD_sub_(FIELD a, FIELD b) {
  bool borrow = 0;
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    FIELD_limb old = a.val[i];
    a.val[i] -= b.val[i] + borrow;
    borrow = borrow ? old <= a.val[i] : old < a.val[i];
  }
  return a;
}

// Normal addition
DEVICE FIELD FIELD_add_(FIELD a, FIELD b) {
  bool carry = 0;
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    FIELD_limb old = a.val[i];
    a.val[i] += b.val[i] + carry;
    carry = carry ? old >= a.val[i] : old > a.val[i];
  }
  return a;
}

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

DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_default(a, b);
}

DEVICE FIELD FIELD_sqr(FIELD a) {
  return FIELD_mul(a, a);
}

DEVICE FIELD FIELD_double(FIELD a) {
  for(uchar i = FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (FIELD_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
  return a;
}

/////////////////////////////////////////////////////
DEVICE POINT_jacobian blst_p1_double_affine(POINT_affine p) {
  /*
      dbl-2009-l from
      http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
  */
  POINT_jacobian out;

  // A = X1^2
  const FIELD A = FIELD_sqr(p.x);
  const FIELD B = FIELD_sqr(p.y);
  FIELD C = FIELD_sqr(B); 

  // D = 2 * ((X1 + B)^2 - A - C)
  FIELD D = FIELD_add(p.x, B);
  D = FIELD_sqr(D); 
  D = FIELD_sub(FIELD_sub(D, A), C); 
  D = FIELD_double(D);

  // E = 3 * A
  const FIELD E = FIELD_add(FIELD_double(A), A);

  // F = E^2
  const FIELD F = FIELD_sqr(E);

  // X3 = F - 2*D
  out.x = FIELD_sub(F, FIELD_double(D));

  // Y3 = E*(D - X3) - 8*C
  C = FIELD_double(C); 
  C = FIELD_double(C); 
  C = FIELD_double(C);
  out.y = FIELD_sub(FIELD_mul(FIELD_sub(D, out.x), E), C);

  // Z3 = 2*Y1 
  out.z = FIELD_double(p.y);

  return out;
}

DEVICE POINT_jacobian blst_p1_double(const POINT_jacobian in) {
  POINT_jacobian out;
    
  if (is_blst_p1_zero(in)) {
    out = in;
  }

  // Z3 = 2*Y1*Z1
  out.z = FIELD_mul(in.y, in.z);
  out.z = FIELD_double(out.z);

  // A = X1^2
  const FIELD a = FIELD_sqr(in.x);

  // B = Y1^2
  const FIELD b = FIELD_sqr(in.y); 

  // C = B^2
  FIELD c = FIELD_sqr(b);

  // D = 2*((X1+B)^2-A-C)
  FIELD d = FIELD_add(in.x, b);
  d = FIELD_sqr(d); 
  d = FIELD_sub(FIELD_sub(d, a), c); 
  d = FIELD_double(d);

  // E = 3 * A
  const FIELD e = FIELD_add(FIELD_double(a), a);

  // F = E^2
  const FIELD f = FIELD_sqr(e);

  // X3 = F - 2*D
  out.x = FIELD_sub(f, FIELD_double(d));

  // Y3 = E*(D - X3) - 8*C
  c = FIELD_double(c); 
  c = FIELD_double(c); 
  c = FIELD_double(c);
  out.y = FIELD_sub(FIELD_mul(FIELD_sub(d, out.x), e), c);

  return out;
}


DEVICE POINT_jacobian blst_p1_add_affines_into_projective(POINT_affine p1, POINT_affine p2) {
  POINT_jacobian out;

  if (is_blst_p1_affine_zero(p2)) {
    out.x = p1.x;
    out.y = p1.y;

    if (is_blst_p1_affine_zero(p1)) {
      out.z = FIELD_ZERO;
    } else {
      out.z = FIELD_ONE;
    }

    return out;
  } else if (is_blst_p1_affine_zero(p1)) {
    out.x = p2.x;
    out.y = p2.y;

    if (is_blst_p1_affine_zero(p2)) {
      out.z = FIELD_ZERO;
    } else {
      out.z = FIELD_ONE;
    }

    return out;
  }

  // mmadd-2007-bl does not support equal values for p1 and p2.
  // If `p1` and `p2` are equal, use the doubling algorithm.
  if(is_blst_fp_eq(p1.x, p2.x) && is_blst_fp_eq(p1.y, p2.y)) {
      return blst_p1_double_affine(p1);
  }

  // H = X2-X1
  FIELD h = FIELD_sub(p2.x, p1.x);
  
  // HH = H^2
  // I = 4*HH
  FIELD i = FIELD_double(h);
  i = FIELD_sqr(i);

  // J = H*I
  FIELD j = FIELD_mul(h,i);

  // r = 2*(Y2-Y1)
  FIELD r = FIELD_sub(p2.y,p1.y);
  r = FIELD_double(r);

  // V = X1*I
  FIELD v = FIELD_mul(p1.x,i);

  // X3 = r^2-J-2*V
  out.x = FIELD_sqr(r);
  out.x = FIELD_sub(out.x, j);
  out.x = FIELD_sub(out.x, FIELD_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  out.y = FIELD_sub(v, out.x);
  out.y = FIELD_mul(out.y, r);

  FIELD y1j = FIELD_mul(p1.y, j);
  out.y = FIELD_sub(out.y, y1j);
  out.y = FIELD_sub(out.y, y1j);

  // Z3 = 2*H
  out.z = FIELD_double(h);

  return out;
}

DEVICE POINT_jacobian blst_p1_add_affine_to_projective(const POINT_jacobian p1, const POINT_affine p2) {
  if (is_blst_p1_affine_zero(p2)) {
    return p1;
  }

  POINT_jacobian out;
  if (is_blst_p1_zero(p1)) {
    out.x = p2.x;
    out.y = p2.y;
    out.z = FIELD_ONE;
    return out;
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
  // Works for all curves.

  // printf("c-t%llu:add:0 %llu,%llu,%llu -> %llu,%llu\n", threadIdx.x, p1->X[0], p1->Y[0], p1->Z[0], p2->X[0], p2->Y[0]);

  // Z1Z1 = Z1^2
  FIELD z1z1 = FIELD_sqr(p1.z);

  // U2 = X2*Z1Z1
  FIELD u2 = FIELD_mul(p2.x, z1z1);

  // S2 = Y2*Z1*Z1Z1
  FIELD s2 = FIELD_mul(p2.y, p1.z);
  s2 = FIELD_mul(s2, z1z1);

  if (is_blst_fp_eq(p1.x, u2) && is_blst_fp_eq(p1.y, s2)) {
    out = blst_p1_double(p1);
    return out;
  }

  // H = U2-X1
  FIELD h = FIELD_sub(u2, p1.x);

  // HH = H^2
  FIELD hh = FIELD_sqr(h);

  // I = 4*HH
  FIELD i = FIELD_double(FIELD_double(hh));

  // J = H*I
  FIELD j = FIELD_mul(h,i);

  // r = 2*(S2-Y1)
  FIELD r = FIELD_sub(s2,p1.y);
  r = FIELD_double(r);

  // V = X1*I
  FIELD v = FIELD_mul(p1.x,i);

  // X3 = r^2 - J - 2*V
  out.x = FIELD_sqr(r);
  out.x = FIELD_sub(out.x,j);
  out.x = FIELD_sub(out.x,FIELD_double(v));
  // out.x = FIELD_sub(out.x,v);

  // Y3 = r*(V-X3)-2*Y1*J
  j = FIELD_mul(p1.y,j);
  j = FIELD_double(j);
  out.y = FIELD_sub(v,out.x);
  out.y = FIELD_mul(out.y,r);
  out.y = FIELD_sub(out.y,j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  out.z = FIELD_add(p1.z,h);
  out.z = FIELD_sqr(out.z);
  out.z = FIELD_sub(out.z,z1z1);
  out.z = FIELD_sub(out.z,hh);

  return out;
}

DEVICE POINT_jacobian blst_p1_add_projective_to_projective(const POINT_jacobian p1, const POINT_jacobian p2) {
  if (is_blst_p1_zero(p2)) {
    return p1;
  }

  if (is_blst_p1_zero(p1)) {
    return p2;
  }

  POINT_jacobian out;
  int p1_is_affine = is_blst_fp_eq(p1.z, FIELD_ONE);
  int p2_is_affine = is_blst_fp_eq(p2.z, FIELD_ONE);

  if (p1_is_affine && p2_is_affine) {
    POINT_affine p1_affine;
    p1_affine.x = p1.x;
    p1_affine.y = p1.y;

    POINT_affine p2_affine;
    p2_affine.x = p2.x;
    p2_affine.y = p2.y;

    out = blst_p1_add_affines_into_projective(p1_affine, p2_affine);
    return out;
  } if (p1_is_affine) {
    POINT_affine p1_affine;
    p1_affine.x = p1.x;
    p1_affine.y = p1.y;

    out = blst_p1_add_affine_to_projective(p2, p1_affine);
    return out;
  } else if (p2_is_affine) {
    POINT_affine p2_affine;
    p2_affine.x = p2.x;
    p2_affine.y = p2.y;

    out = blst_p1_add_affine_to_projective(p1, p2_affine);
    return out;
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
  // Works for all curves.

  // printf("c-t%llu:add:0 %llu,%llu,%llu -> %llu,%llu\n", threadIdx.x, p1->X[0], p1->Y[0], p1->Z[0], p2->X[0], p2->Y[0]);

  // Z1Z1 = Z1^2
  FIELD z1z1 = FIELD_sqr(p1.z);

  // Z2Z2 = Z2^2
  FIELD z2z2 = FIELD_sqr(p2.z);

  // U1 = X1*Z2Z2
  FIELD u1 = FIELD_mul(p1.x, z2z2);

  // U2 = X2*Z1Z1
  FIELD u2 = FIELD_mul(p2.x, z1z1);

  // S1 = Y1*Z2*Z2Z2
  FIELD s1 = FIELD_mul(p1.y, p2.z);
  s1 = FIELD_mul(s1, z2z2);

  // S2 = Y2*Z1*Z1Z1
  FIELD s2 = FIELD_mul(p2.y, p1.z);
  s2 = FIELD_mul(s2, z1z1);

  // H = U2-U1
  FIELD h = FIELD_sub(u2,u1);

  // HH = H^2
  // FIELD hh = FIELD_sqr(h);

  // I = 4*HH
  FIELD i = FIELD_double(h);
  i = FIELD_sqr(i);

  // J = H*I
  FIELD j = FIELD_mul(h,i);

  // r = 2*(S2-S1)
  FIELD r = FIELD_sub(s2, s1);
  r = FIELD_double(r);

  // V = U1*I
  FIELD v = FIELD_mul(u1,i);

  // X3 = r^2 - J - 2*V
  out.x = FIELD_sqr(r);
  out.x = FIELD_sub(out.x,j);
  out.x = FIELD_sub(out.x,FIELD_double(v));
  // out.x = FIELD_sub(out.x,v);

  // Y3 = r*(V-X3)-2*S1*J
  j = FIELD_mul(s1,j);
  j = FIELD_double(j);
  out.y = FIELD_sub(v,out.x);
  out.y = FIELD_mul(out.y,r);
  out.y = FIELD_sub(out.y,j);

  //  Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H
  out.z = FIELD_add(p1.z,p2.z);
  out.z = FIELD_sqr(out.z);
  out.z = FIELD_sub(out.z,z1z1);
  out.z = FIELD_sub(out.z,z2z2);
  out.z = FIELD_mul(out.z,h);

  return out;
}


//////////////////////////////////////////////////////////////////////////////////////////////
KERNEL void msm6_pixel(GLOBAL POINT_jacobian* bucket_lists, const GLOBAL POINT_affine* bases_in, const GLOBAL blst_scalar* scalars, const GLOBAL uint* window_lengths, const uint window_count) {
  uint threadIdxx = get_local_id(0);
  uint blockIdxx = get_group_id(0);
  
  uint index = threadIdxx / 64;
  size_t shift = threadIdxx - (index * 64);
  ulong mask = (ulong) 1 << (ulong) shift;

  POINT_jacobian bucket = POINT_ZERO;

  uint window_start = WINDOW_SIZE * blockIdxx;
  uint window_end = window_start + window_lengths[blockIdxx];

  uint activated_bases[128];
  uint activated_base_index = 0;

  uint i;
  for (i = window_start; i < window_end; ++i) {
    ulong bit = (scalars[i].val[index] & mask);
    if (bit == 0) {
        continue;
    }
    activated_bases[activated_base_index++] = i;
  }
  BARRIER_LOCAL();

  i = 0;
  // for (; i < (activated_base_index / 2 * 2); i += 2) {
  //   POINT_jacobian intermediate = blst_p1_add_affines_into_projective(bases_in[activated_bases[i]], bases_in[activated_bases[i + 1]]);
  //   bucket = blst_p1_add_projective_to_projective(bucket, intermediate);
  // }
  for (; i < activated_base_index; ++i) {
    bucket = blst_p1_add_affine_to_projective(bucket, bases_in[activated_bases[i]]);
  }

  bucket_lists[threadIdxx * window_count + blockIdxx] = bucket;
  BARRIER_LOCAL();
}

KERNEL void msm6_collapse_rows(GLOBAL POINT_jacobian* target, const GLOBAL POINT_jacobian* bucket_lists, const uint window_count) {
  uint threadIdxx = get_local_id(0);
  uint blockIdxx = get_group_id(0);

  uint base = threadIdxx * window_count;
  uint term = base + window_count;

  POINT_jacobian temp_target = bucket_lists[base];

  for (uint i = base + 1; i < term; ++i) {
    temp_target = blst_p1_add_projective_to_projective(temp_target, bucket_lists[i]);
  }

  target[threadIdxx] = temp_target;
  BARRIER_LOCAL();
}


