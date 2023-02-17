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

typedef struct { FIELD X, Y; } blst_p1_affine;
typedef struct { FIELD X, Y, Z; } blst_p1;
typedef struct { FIELD X, Y, ZZ, ZZZ; } blst_p1_ext;

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

CONSTANT FIELD FIELD_ZERO = {{
  0, 0,
  0, 0,
  0, 0
  }};

CONSTANT blst_p1 BLS12_377_ZERO_PROJECTIVE = {{
  {0},
  {FIELD_ONE},
  {0}
}};

CONSTANT uint WINDOW_SIZE = 128;

/////////////////////////////////////////////////////

KERNEL void msm6_pixel(GLOBAL blst_p1* bucket_lists, GLOBAL blst_p1_affine* bases_in, GLOBAL blst_scalar* scalars, GLOBAL uint* window_lengths, uint window_count) {
  uint index = get_local_id(0) / 64;
  size_t shift = get_local_id(0) - (index * 64);
  ulong mask = (ulong) 1 << (ulong) shift;

  blst_p1 bucket = BLS12_377_ZERO_PROJECTIVE;

  uint window_start = WINDOW_SIZE * get_group_id(0);
  uint window_end = window_start + window_lengths[get_group_id(0)];

  LOCAL uint* activated_bases;

  // uint activated_bases[WINDOW_SIZE];

}

// KERNEL void FIELD_radix_fft(GLOBAL FIELD* x, // Source buffer
//                       GLOBAL FIELD* y, // Destination buffer
//                       GLOBAL FIELD* pq, // Precalculated twiddle factors
//                       GLOBAL FIELD* omegas, // [omega, omega^2, omega^4, ...]
//                       LOCAL FIELD* u_arg, // Local buffer to store intermediary values
//                       uint n, // Number of elements
//                       uint lgp, // Log2 of `p` (Read more in the link above)
//                       uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
//                       uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
// {}


////////////////////////
// typedef unsigned int uint32_t;
// typedef unsigned long limb_t;

// #define LIMB_T_BITS    64
// #define TO_LIMB_T(limb64)     limb64
// #define NLIMBS(bits)   (bits/LIMB_T_BITS)

// typedef limb_t blst_fp[NLIMBS(256)];
// typedef struct { blst_fp X, Y, Z; } blst_p1;

// KERNEL void msm6_collapse_rows(uint32_t *target, uint32_t *bucket_lists, uint32_t window_count) {
    // blst_p1 temp_target;
    // uint32_t base = threadIdx.x * window_count;
    // uint32_t term = base + window_count;
    // memcpy(&temp_target, &bucket_lists[base], sizeof(blst_p1));

    // for (uint32_t i = base + 1; i < term; ++i) {
    //     blst_p1_add_projective_to_projective(&temp_target, &temp_target, &bucket_lists[i]);
    // }
    
    // memcpy(&target[threadIdx.x], &temp_target, sizeof(blst_p1));
// }

// KERNEL void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
//     for (uint i = 0; i < num; i++) {
//       result[i] = a[i] + b[i];
//     }
// }