#include <stdio.h>
#include <stdlib.h>


char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    int size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    while(line[curr-1]!='\n'){
        size *= 2;
        line = realloc(line, size*sizeof(char));
        if(!line) error("Malloc");
        fgets(&line[curr], size-curr, fp);
        curr = strlen(line);
    }
    line[curr-1] = '\0';

    return line;
}

int main(void){
	char *cfg = "cfg/yolov3.cfg";
	FILE *fp = fopen(cfg, "r");
	char *line;
	line = fgetl(fp);
	printf("%s\n", line);
	printf("%d\n", strlen(line));
}
