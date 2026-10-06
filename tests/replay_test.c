/* Host-only I/O. Core embedded/ has no filesystem or OS dependencies. */
#include "fall_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static void array(FILE *f, const FallReal *a, unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) fprintf(f,",%.17g",(double)a[i]);
}
int main(int argc, char **argv) {
    FILE *input, *streams, *events;
    FallDetector d;
    char line[1024];
    FallOperatingPoint point=FALL_BALANCED_REFERENCE;
    if(argc==2 && strcmp(argv[1],"--size")==0) {
        printf("%zu\n",sizeof(FallDetector)); return 0;
    }
    if(argc!=5) {
        fprintf(stderr,"Usage: replay_test raw.txt streams.csv events.csv balanced|sensitivity\n"); return 2;
    }
    if(strcmp(argv[4],"sensitivity")==0) point=FALL_SENSITIVITY_CANDIDATE;
    else if(strcmp(argv[4],"balanced")!=0) return 2;
    input=fopen(argv[1],"r"); streams=fopen(argv[2],"w"); events=fopen(argv[3],"w");
    if(!input || !streams || !events) { fprintf(stderr,"Cannot open replay files\n"); return 2; }
    fall_detector_init_operating_point(&d,point);
    fprintf(streams,"index,fx,fy,fz,ax,ay,az,gx,gy,gz,ux,uy,uz,bx,by,bz,m,h,p,jerk,high,edge,trigger\n");
    fprintf(events,"trigger,start,end,decision,ga_C2,jerk_abs_mean,ga_parallel_peak,z,label\n");
    while(fgets(line,sizeof(line),input)) {
        int x,y,z;
        FallResult r;
        FallTrace *t=&d.trace;
        /* SisFall first three columns are integer ADXL345 counts. */
        if(line[0]=='\n' || line[0]=='\r') continue;
        if(sscanf(line," %d , %d , %d",&x,&y,&z)!=3 ||
           x<INT16_MIN || x>INT16_MAX || y<INT16_MIN || y>INT16_MAX || z<INT16_MIN || z>INT16_MAX) {
            fprintf(stderr,"Invalid counts at sample %lld\n",(long long)d.next_index); return 3;
        }
        r=fall_detector_push_counts(&d,(int16_t)x,(int16_t)y,(int16_t)z);
        fprintf(streams,"%lld",(long long)(d.next_index-1));
        array(streams,t->filtered,3); array(streams,t->acceleration,3);
        array(streams,t->gravity,3); array(streams,t->unit,3); array(streams,t->residual,3);
        fprintf(streams,",%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n",
            (double)t->magnitude,(double)t->perpendicular,(double)t->parallel,(double)t->jerk,
            t->high,t->rising_edge,t->accepted_trigger);
        if(r!=FALL_NO_DECISION) {
            FallEvent *e=&d.event;
            fprintf(events,"%lld,%lld,%lld,%lld",(long long)e->trigger,(long long)e->start,
                    (long long)e->end,(long long)e->decision);
            array(events,e->features,3);
            fprintf(events,",%.17g,%d\n",(double)e->score,r==FALL_DETECTED);
        }
    }
    if(ferror(input)) return 4;
    fclose(input);
    if(fclose(streams)!=0 || fclose(events)!=0) return 4;
    return 0; /* no EOF flushing: an unfinished event is discarded */
}
