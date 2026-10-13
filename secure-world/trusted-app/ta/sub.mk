global-incdirs-y += include
srcs-y += rawtfnet_ta.c
srcs-y += rawtfnet_model.c
srcs-y += ta_math.c

cflags-rawtfnet_model.c-y += -Wno-unused-variable -Wno-unused-but-set-variable
