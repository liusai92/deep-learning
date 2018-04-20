/*
 * author: liusai
 * email: liusai.kepler@gmail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct
{
    float** imdt; //image data
    int width;
    int height;
}
Image;

typedef struct
{
    float** ftdt; //filter data
    int size; // filter length with same width and height
}
Filter;

/*
 * conv kernel for conv2d, compute each element of out image for conv2d.
 */
static float conv_kernel(const Image image, const Filter filter, int startw, int starth)
{
    float sum = 0.0f;
    for (int i=starth; i<(starth+filter.size); i++) {
        for (int j=startw; j<(startw+filter.size); j++) {
            sum += image.imdt[i][j] * filter.ftdt[i-starth][j-startw];
        }
    }
    return sum;
}

/*
 * default padding as "valid", refer https://www.tensorflow.org/versions/r1.5/api_guides/python/nn
 */
void conv2d(const Image im_in, Image im_out, const Filter filter)
{
    int startw,starth; //index in original image;

    // compute element by element
    for (int i=0; i<im_out.height; i++)
    {
        starth = i*strides;
        for (int j=0; j<im_out.width; j++)
        {
            startw = j*strides;
            im_out.imdt[i][j] = conv_kernel(im_in, filter, startw, starth);
        }
    }
} 

/*
 * back propagation of conv2d, S: strides
 */
float bconv2d(const Image im_in, Image dlim, int k, int l, int S)
{
    float sum = 0.0f;
    for(int m=0; m<dlim.height; m++)
    {
        for(int n=0; n<dlim.width; n++)
            sum += dlim.imdt[m][n] * im_in.imdt[k+m*S][l+n*S]
    }
    return sum;
}

// Floating point random from 0.0 - 1.0.
static float frand()
{
    return rand() / (float) RAND_MAX;
}

static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}


/* 
 * random initialize a filter
 */
Filter nfilter(int size)
{
    Filter ft;
    ft.size = size;
    //allocate memory on heap
    float** ftdt = (float**) malloc(size * sizeof(float*));
    for(int i=0; i<size; i++)
        ftdt[i] = (float*) malloc(size * sizeof(float));
    //randomalize ftdt
    for(int i=0; i<size; i++)
        for(int j=0; j<size; j++)
            ftdt[i][j] = frand();
    ft.ftdt = ftdt;
    return ft;
}


/*
 * read one line from file, save to line[2048]
 */
static void readln(FILE* const file, char* line)
{
    int ch = EOF;
    int reads = 0;
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
    }
    line[reads] = '\0';
}

/*
 * parse one line, save data to im_data, return label
 * in_width: input image's width; in_height: input image's height
 * nips: number of fields of one line from file
 */
static int parse(char* line, float** im_data, int in_width, int in_height)
{
    int nips = in_width * in_height;
    char* il[nips];

    int label = atoi(strtok(line, ","));

    for (int i=0; i<nips; i++)
        il[i] = strtok(NULL, ",");

    int ild = 0;
    for (int i=0; i<in_height; i++)
        for (int j=0; j<in_width; j++)
            im_data[i][j] = atof(il[ild++]) > 0.0 ? 1.0 : 0.0;

    return label;
}

/*
 * initialize increment to 0
 */
void init(float** icr, int colos, int rows)
{
    for(int i=0; i<rows; i++)
        for(int j=0; j<colos; j++)
            icr[i][j] = 0.0f;
}

/*
 * flatten out image to hidden layer
 */
void flatten(Image im_out, float* hid)
{
    if(im_out.height * im_out.width != sizeof(hid))
    {
        fprintf(stderr, "flatten: the length of hidden layer should\
                be im_out.width * im_out.height\n");
        exit(EXIT_FAILURE);
    }
    int ih = 0;
    for(int i=0; i<im_out.height; i++)
        for(int j=0; j<im_out.width; j++)
            hid[ih++] = im_out.imdt[i][j];
}

/*
 * unflatten gradient of hidden layer to gradient of out image
 */
void unflatten(Image dlim, float* dlh)
{
    if(dlim.height * dlim.width != sizeof(dlh))
    {
        fprintf(stderr, "unflatten: the length of dlh should be\
                        dlim.height * dlim.width\n");
        exit(EXIT_FAILURE);
    }
    int ih=0;
    for(int i=0; i<dlim.height; i++)
        for(int j=0; j<dlim.width; j++)
            dlim.imdt[i][j] = dlh[ih++];
}

/*
 * compute softmax of logit, output to prob.
 */
void softmax(float* logit, float* prob, int num_classes)
{
    float sum = 0.0f;
    for(int i=0; i<num_classes; i++)
    {
        prob[i] = (float) exp(logit[i]);
        sum += prob[i];
    }
    for(int i=0; i<num_classes; i++)
        prob[i] /= sum;
}

/*
 * compute cross_entropy
 */
float cross_entropy(float* prob, int label)
{
    return (float) -1.0*log(prob[label-1]);
}

/*
 * activated function
 */
float relu(float inp)
{
    if(inp>0)
        return inp;
    else
        return 0.0f;
}

/*
 * gradient of loss w.r.t prob
 */
void dl2p(float* dlp, float* prob, int label, int num_classes)
{
    for(int i=0; i<num_classes; i++)
        dlp[i] = 0.0f;
    dlp[label-1] = -1.0 / prob[label-1];
}

/* 
 * gradient of loss w.r.t logit, refer formula to README.md
 * dlp: gradient of loss w.r.t prob, should be the return of dl2p()
 */
void dl2l(float* dll, float* logit, float* prob, float* dlp, int label, int num_classes)
{
    for(int i=0; i<num_classes; i++)
    {
        if(i!=label-1)
            dll[i] = -prob[label-1] * prob[i];
        else
        {
            for(int j=0; j<num_classes; j++)
                if(j!=label-1) dll[i] -= prob[label-1]*prob[j];
        }
    }
    for(int i=0; i<num_classes; i++)
        dll[i] *= dlp[label-1];
}

//save filter: ft and weights: w
void save(char* path, Filter ft, float** w, int num_classes, int lhid)
{
    FILE* f = fopen(path, "w");
    fprintf(f, "%d\n", ft.size);
    for(int i=0; i<ft.size; i++)
        for(int j=0; j<ft.size; j++)
            fprintf(f, "%f\n", ft.ftdt[i][j]);
    fprintf(f, "%d\n", num_classes);
    fprintf(f, "%d\n", lhid);
    for(int i=0; i<num_classes; i++)
        for(int j=0; j<lhid; j++)
            fprintf(f, "%f\n", w[i][j]);
}

// free Image
static void ifree(const Image image)
{
    for(int row = 0; row < image.height; row++)
        free(image.imdt[row]);
    free(image.imdt);
}

//free Filter
static void ffree(const Filter filter)
{
    for (int row = 0; row < filter.size; row++)
        free(filter.ftdt[row]);
    free(filter.ftdt);
}

int main()
{
    srand(time(0));
    int batch_size = 128;
    int num_steps = 500; //training steps, train one batch at every step
    int num_classes = 10;
    int width, height = 28, 28; //image width & image height
    int strides = 2;
    float lr = 0.1f; //learning rate
    FILE* f = fopen("/path/to/data", "r");

    //declare variable for future use
    char line[2048];
    float** data[batch_size]; //a batch of im_in.data
    int labels[batch_size];   //a batch of labels
    //allocate memory for data on heap
    for(int i=0; i<batch_size; i++)
        data[i] = new2d(height, width);
    Image im_in = {.width=width, .height=height}; //input image
    Filter ft = nfilter(5); //initialize a 5*5 filter
    //compute out image width and height
    int out_width = (int) ceil((float)(image.width-filter.size+1)/(float)(strides));
    int out_height = (int) ceil((float)(image.height-filter.size+1)/(float)(strides));
    //allocate memory for im_out data on heap
    Image im_out = {new2d(out_height, out_width), out_width, out_height}; //output image
    int lhid = out_width * out_height; //hidden layer length
    float hid[lhid]; //hidden layer
    float w[num_classes][lhid]; //weights between hidden layer and output layer
    float b = frand();//bias
    float logit[num_classes];
    float prob[num_classes];
    float loss; //loss value
    //back propagation parameters
    float dlp[num_classes]; //gradient of loss w.r.t prob
    float dll[num_classes]; //gradient of loss w.r.t logit
    float dw[num_classes][lhid]; //weights increment
    float dlh[lhid]; //gradient of loss w.r.t hidden layer
    Image dlim = {new2d(out_height, out_width), out_width, out_height}; //gradient for out image
    float dft[ft.size][ft.size]; //filter increment

    //begin training
    for(int step=1; step<num_steps+1; step++)
    {
        //generate batch
        for(int bidx=0; bidx<batch_size; bidx++)
        {
            readln(f, line);
            if(line[0]=='\0'){
                fclose(f);
                FILE* f = fopen("/path/to/data", "r");
                readln(f, line);
            }
            labels[bidx] = parse(line, data[bidx], width, height);
        }
        //initialize increment
        init(dft, ft.size, ft.size);
        init(dw, lhid, num_classes);
        loss = 0.0f;
        //batch training
        for(int bidx=0; bidx<batch_size; bidx++)
        {
            //forward propagation
            im_in.imdt = data[bidx];
            conv2d(im_in, im_out, ft);
            flatten(im_out, hid);
            for(int i=0; i<num_classes; i++)
            {
                logit[i] = 0.0f;
                for(int j=0; j<lhid; j++)
                    logit[i] += hid[j] * w[i][j];
                logit[i] += b;
                logit[i] = relu(logit[i]);
            }
            softmax(logit, prob, num_classes);
            loss += cross_entropy(prob, labels[bidx])/batch_size;
            //back propagation
            dl2p(dlp, prob, labels[bidx], num_classes);
            dl2l(dll, logit, prob, dlp, labels[bidx], num_classes); 
            //update weights increment
            for(int i=0; i<num_classes; i++)
            {
                for(int j=0; j<lhid; j++)
                    dw[i][j] = dll[i] * (logit[i] == 0.0 ? 0 : hid[j])/batch_size;
            }
            for(int j=0; j<lhid; j++)
            {
                dlh[j] = 0.0f;
                for(int i=0; i<num_classes; i++)
                    dlh[j] += dll[i] * (logit[i] == 0.0 ? 0 : w[i][j]);
            }
            unflatten(dlim, dlh);
            for(int i=0; i<ft.size; i++)
            {
                for(int j=0; j<ft.size; j++)
                    dft[i][j] = bconv2d(im_in, dlim, i, j, strides)/batch_size;
            }
        }
        //update ft and w after one batch
        for(int i=0; i<ft.size; i++)
            for(int j=0; j<ft.size; j++)
                ft.ftdt[i][j] -= lr * dft[i][j];
        for(int i=0; i<num_classes; i++)
            for(int j=0; j<lhid; j++)
                w[i][j] -= lr* dw[i][j];
        printf("Step: %.4d, loss: %.6f\n", step, loss);
        save("/path/to/savefile", dft, w, num_classes, lhid);
    }

    //free heap variable, close file
    fclose(f);
    ifree(im_out);
    ifree(dlim);
    ffree(ft);
    for(int i=0; i<batch_size; i++)
    {
        for(int j=0; j<height; j++)
            free(data[i][j]);
        free(data[i]);
    }
    return 0;
}
