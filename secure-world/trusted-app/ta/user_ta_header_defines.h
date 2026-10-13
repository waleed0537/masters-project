#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

#include <rawtfnet_ta.h>

#define TA_UUID				TA_RAWTFNET_UUID

/* Model code+weights are large; give the TA generous memory. */
#define TA_FLAGS			(TA_FLAG_EXEC_DDR | TA_FLAG_SINGLE_INSTANCE | \
					 TA_FLAG_MULTI_SESSION)
#define TA_STACK_SIZE			(256 * 1024)
#define TA_DATA_SIZE			(4 * 1024 * 1024)

#define TA_CURRENT_TA_EXT_PROPERTIES \
	{ "gp.ta.description", USER_TA_PROP_TYPE_STRING, "RawTFNet deepfake TA" }, \
	{ "gp.ta.version", USER_TA_PROP_TYPE_U32, &(const uint32_t){ 0x0010 } }

#endif /* USER_TA_HEADER_DEFINES_H */
