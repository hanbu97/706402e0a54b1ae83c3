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
  
CONSTANT FIELD FIELD_ZERO = { { 0, 0, 0, 0, 0, 0 } };
CONSTANT uint WINDOW_SIZE = 128;

/////////////////////////////////////////////////////

KERNEL void msm6_pixel(GLOBAL POINT_jacobian* bucket_lists, GLOBAL POINT_affine* bases_in, GLOBAL blst_scalar* scalars, GLOBAL uint* window_lengths, uint window_count) {
  uint index = get_local_id(0) / 64;
  size_t shift = get_local_id(0) - (index * 64);
  ulong mask = (ulong) 1 << (ulong) shift;

  const POINT_jacobian bucket = POINT_ZERO;

  uint window_start = WINDOW_SIZE * get_group_id(0);
  uint window_end = window_start + window_lengths[get_group_id(0)];

  LOCAL uint* activated_bases;
  uint activated_base_index = 0;

  uint i;
  for (i = window_start; i < window_end; ++i) {
    // ulong bit = (scalars[i][index] & mask);
    // if (bit == 0) {
    //     continue;
    // }
    activated_bases[activated_base_index++] = i;
  }

}

