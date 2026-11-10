// 逆順の文章の BWT を作る: R = reverse(T); BWT[i] = R[SA[i]-1]. 出力: bwt.bin (uint16)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "libsais/include/libsais.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET); long n=nb/2;
  std::vector<uint16_t> t(n); fread(t.data(),2,n,f); fclose(f);
  std::vector<int32_t> R(n), SA(n);
  for(long i=0;i<n;i++) R[i]=t[n-1-i];
  int32_t rc=libsais_int(R.data(),SA.data(),(int32_t)n,32768,0);
  if(rc!=0){fprintf(stderr,"libsais rc=%d\n",rc);return 1;}
  std::vector<uint16_t> bwt(n);
  for(long i=0;i<n;i++) bwt[i]= SA[i]==0 ? 0 : (uint16_t)R[SA[i]-1];
  FILE*g=fopen(argv[2],"wb"); fwrite(bwt.data(),2,n,g); fclose(g);
  long Z=-1; for(long i=0;i<n;i++) if(SA[i]==0){Z=i;break;}
  FILE*h=fopen(argv[3],"w"); fprintf(h,"{\"n\": %ld, \"Z\": %ld, \"first\": %d}\n",n,Z,(int)t[0]); fclose(h);
  printf("n=%ld Z=%ld ok\n",n,Z); return 0; }
