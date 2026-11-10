FM-index prototype for count index (all orders from one index).
- mkbwt.cpp: SA/BWT of reversed token stream via libsais (https://github.com/IlyaGrebnov/libsais); build: gcc -O2 -c libsais/src/libsais.c; g++ mkbwt.cpp libsais.o
- fmq.cpp: flat wavelet matrix + backward search; outputs N, distinct followers, count(w) per order (1-6,8). Needs sentinel row index Z (row where SA==0) and the extra end-marker row.
- wm_size.py: RRR(31) compressed-size estimate of the wavelet matrix.
- cmp.py: compares with engine tables (orders 3-8 exact on 2000 positions; orders 1-2 differ only by pad zeros).
Result (28M tokens): ~1.24 B/token estimated compressed, 1.15 ms/position (7 orders, uncompressed).
