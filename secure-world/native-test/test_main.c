#include <stdio.h>
#include <stdlib.h>

void model(const float audio[restrict 1][64600], float logits[restrict 2]);

static float in[1][64600];
static float out[2];

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s input.bin\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 1; }
    size_t n = fread(in, sizeof(float), 64600, f);
    fclose(f);
    if (n != 64600) { printf("expected 64600 floats, got %zu\n", n); return 1; }

    model(in, out);

    printf("logits: %f %f\n", out[0], out[1]);
    printf("verdict: %s\n", out[1] > 0.0f ? "BONAFIDE (real)" : "SPOOF (fake)");
    return 0;
}
