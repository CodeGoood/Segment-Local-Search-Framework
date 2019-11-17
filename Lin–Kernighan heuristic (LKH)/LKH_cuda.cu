#include <iostream>
#include <float.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cmath>

__host__ __device__ long long int iDivUp(long long int a, long long int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

#define CORE ((int)3038)
#define the CUDA_API_PER_THREAD_DEFAULT_STREAM


#define gpuErrchk(ans) { HANDLE_ERROR((ans), __FILE__, __LINE__); }
inline void HANDLE_ERROR(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        printf( "ERROR: %s in %s at line %d\n", cudaGetErrorString( code ),file, line );
        exit(0);
    }
}

clock_t sub_start, sub_end;
int DEVICE_ID;

typedef struct point 
{
    float x;
    float y;
} Point;

class Edge
{
    private:
        int endPoint1;
        int endPoint2;

    public:
        __device__ __host__ Edge(int a, int b);
        __device__ void set(int a);
        __device__ int get1();
        __device__ int get2();
        __device__ int compareTo(Edge e2);
        __device__ bool equals(Edge e2);
};

Edge::Edge(int a, int b) 
{
    endPoint1 = a > b? a:b;
    endPoint2 = a > b? b:a;
}

__device__ void Edge::set(int a)
{
    endPoint1 = a;
    endPoint2 = a;
}

__device__ int Edge::get1() 
{
    return endPoint1;
}

__device__ int Edge::get2() 
{
    return endPoint2;
}

__device__ int Edge::compareTo(Edge e2) 
{
    if(get1() < e2.get1() || get1() == e2.get1() && get2() < e2.get2()) {
        return -1;
    } else if (equals(e2)) {
        return 0;
    } else {
        return 1;
    }

}

__device__ bool Edge::equals(Edge e2) {
    if(e2.get1() == -99) return false;
    return (get1() == e2.get1()) && (get2() == e2.get2());
}

int SegmentLength;
int NoImprovedNumber;
double power = 0;
double sita = 0;
FILE *fp, *ws_fp, *pop_fp, *runtime_fp;

int *fullTour;
int *tmp_fullTour;

// The distance table
int **distanceTable;

int countcity(char *fn)
{
    int tmp;
    float a,b;
    FILE *fc;
    if ((fc=fopen(fn,"r"))==NULL)
    {
        printf("open file %s error\n",fn);
        exit(0);
    }

    int i = 0;
    while(fscanf(fc, "%d %f %f", &tmp, &a, &b) != EOF) {i++;}
    return i;
}

__device__ int getDistance2node(int n1, int n2, int *tour, Point *d_coordinates, long long int *edgeCount) {
    edgeCount[0] += 1;
    int i = tour[n1];
    int j = tour[n2];
    double q1 = d_coordinates[j].x-d_coordinates[i].x;
    double q2 = d_coordinates[j].y-d_coordinates[i].y;
    
    return (int) sqrt( q1*q1+q2*q2 );
}


__device__ int getDistance2nodeActualNode(int n1, int n2, Point *d_coordinates) {
    //edgeCount[0] += 1;
    double q1 = d_coordinates[n1].x-d_coordinates[n2].x;
    double q2 = d_coordinates[n1].y-d_coordinates[n2].y;
    
    return (int) sqrt( q1*q1+q2*q2 );
}

int getDistance2nodeActualNodeHost(int n1, int n2, Point *coordinates) {
    double q1 = coordinates[n1].x-coordinates[n2].x;
    double q2 = coordinates[n1].y-coordinates[n2].y;
    
    return (int) sqrt( q1*q1+q2*q2 );
}

void readcity_coords(char *fn, Point *coordinates)
{
    int tmp;
    FILE *fc;
    fc=fopen(fn,"r");

    int i = 0;
    while(fscanf(fc, "%d %f %f", &tmp, &coordinates[i].x, &coordinates[i].y) != EOF) {i++;}
}

void initDistanceTable(int *dimension, Point *coordinates) 
{
    distanceTable = (int**) malloc( dimension[0] * sizeof(int*));
    for(int i = 0; i < dimension[0]; i++)
        distanceTable[i] = (int*) malloc(dimension[0] * sizeof(int));

    for(int i = 0; i < dimension[0]-1; ++i) 
    {
        for(int j = i + 1; j < dimension[0]; ++j) 
        {   
            long long int q1 = coordinates[j].x-coordinates[i].x;
            long long int q2 = coordinates[j].y-coordinates[i].y;

            distanceTable[i][j] = (int) sqrt( q1*q1+q2*q2 );
            distanceTable[j][i] = distanceTable[i][j];
        }
    }

}

__device__ int getNextIdx(int index, int *size) {
    return (index + 1) % size[0];
}

__device__ int getPreviousIdx(int index, int *size) {
    return index == 0 ? size[0] - 1 : index - 1;
}

void createRandomTour(int *dimension)
{
    std::vector<int> myvector;
    for (int k=0; k < dimension[0]; k++) myvector.push_back(k); // 1 2 3 4 5 6 7 8 9

    random_shuffle ( myvector.begin(), myvector.end() );
    for (int k=0; k < dimension[0]; k++)
        fullTour[k] = myvector[k];
}

double getDistance(int *dimension) {
    double sum = 0;

    for(int i = 0; i < dimension[0]; i++) {
        int a = fullTour[i];                  // <->
        int b = fullTour[(i+1)%dimension[0]];    // <->
        sum += distanceTable[a][b];
    }

    return sum;
}


__device__ bool isTour(int *tour, int *tourOT_size, int *size) {
    if (tourOT_size[0] != size[0]) {
        return false;
    }

    for (int i = 0; i < size[0] - 1; ++i) {
        for (int j = i + 1; j < size[0]; ++j) {
            if (tour[i] == tour[j]) {
                return false;
            }
        }
    }

    return true;
}

__device__ void createTourFromEdgesOnly(Edge *currentEdges, int s, int *tour_multi, int currentEdge_size)
{
    int i = 0;
    int last = -1;

    for(; i < currentEdge_size; ++i)
    {
        if(currentEdges[i].get1() != -99)
        {
            tour_multi[0] = currentEdges[i].get1();
            tour_multi[1] = currentEdges[i].get2();
            last = tour_multi[1];
            break;
        }
    }

    currentEdges[i].set(-99);

    int k = 2;
    while (true) {

        int j = 0;
        for(; j < currentEdge_size; ++j) {
            if(currentEdges[j].get1() != -99 && currentEdges[j].get1() == last) {
                last = currentEdges[j].get2();
                break;
            } else if(currentEdges[j].get1() != -99 && currentEdges[j].get2() == last) {
                last = currentEdges[j].get1();
                break;
            }
        }

        if (j == currentEdge_size) break;

        // Remove new edge
        currentEdges[j].set(-99);
        if (k >= s) break;
        tour_multi[k] = last;
        k++;
    }

}

__device__ void createTourFromEdgesOT(Edge *currentEdges, int s, int *tour_multi, int currentEdge_size, int *tourOT_size) 
{
    int i = 0;
    int last = -1;

    for(; i < currentEdge_size; ++i) 
    {
        if(currentEdges[i].get1() != -99) 
        {
            tour_multi[0] = currentEdges[i].get1();
            tour_multi[1] = currentEdges[i].get2();
            last = tour_multi[1];
            break;
        }
    }
    currentEdges[i].set(-99);

    int k = 2;
    while (true) {

        int j = 0;
        for(; j < currentEdge_size; ++j) {
            if(currentEdges[j].get1() != -99 && currentEdges[j].get1() == last) {
                last = currentEdges[j].get2();
                break;
            } else if(currentEdges[j].get1() != -99 && currentEdges[j].get2() == last) {
                last = currentEdges[j].get1();
                break;
            }
        }

        // If the list is empty
        if (j == currentEdge_size) break;

        // Remove new edge
        currentEdges[j].set(-99);
        if (k >= s) break;
        tour_multi[k] = last;
        k++;
    }
    tourOT_size[0] = s;

}

__device__ void constructNewTour_Only(int *tIndex, Edge *currentEdges, int *tour, int tIndex_size, int *size)
{
    int currentEdges_size = size[0];

    for (int i = 0; i < size[0]; ++i)
        currentEdges[i] = Edge(tour[i], tour[(i + 1) % size[0]]);

    int s = currentEdges_size;

    // Remove Xs
    for(int i=1; i< tIndex_size - 2; i += 2)
    {
        for (int j = 0; j < currentEdges_size; ++j) 
        {
            Edge m = currentEdges[j];//.get(j);
            if (Edge(tour[tIndex[i]], tour[tIndex[i + 1]]).equals(m))
            {
                s--;
                currentEdges[j].set(-99);
                break;
            }
        }
    }

    // Add Ys
    for(int i=2; i< tIndex_size - 1; i += 2)
    {
        s++;
        currentEdges[currentEdges_size] = Edge(tour[tIndex[i]], tour[tIndex[i + 1]]);
        currentEdges_size++;
    }

    createTourFromEdgesOnly(currentEdges, s, tour, currentEdges_size);
}

__device__ void constructNewTour_OneTwo(int *tIndex, Edge *currentEdges, int *tourOneTwo, int *tour, int *tourOT_size, int tIndex_size, int *size)
{
    int currentEdges_size = size[0];

    for (int i = 0; i < size[0]; ++i)
        currentEdges[i] = Edge(tour[i], tour[(i + 1) % size[0]]);

    int s = currentEdges_size;

    // Remove Xs
    for(int i=1; i< tIndex_size - 2; i += 2)
    {
        for (int j = 0; j < currentEdges_size; ++j) 
        {
            Edge m = currentEdges[j];//.get(j);
            if (Edge(tour[tIndex[i]], tour[tIndex[i + 1]]).equals(m))
            {
                s--;
                currentEdges[j].set(-99);
                break;
            }
        }
    }

    // Add Ys
    for(int i=2; i< tIndex_size - 1; i += 2)
    {
        s++;
        currentEdges[currentEdges_size] = Edge(tour[tIndex[i]], tour[tIndex[i + 1]]);
        currentEdges_size++;
    }

    createTourFromEdgesOT(currentEdges, s, tourOneTwo, currentEdges_size, tourOT_size);

}

__device__ bool isDisjunctive(int *tIndex, int x, int y, int tIndex_size) {
    if (x == y) return false;

    for (int i = 0; i < tIndex_size - 1; i++) {
        if (tIndex[i] == x && tIndex[i+1] == y) return false;
        if (tIndex[i] == y && tIndex[i+1] == x) return false;
    }
    return true;
}


__device__ void getTPrime(int *tIndex, int k, Edge *currentEdges, int *tour, int *size)
{
    constructNewTour_Only(tIndex, currentEdges,tour, k+2, size);
}

__device__ bool isConnected( int *tIndex, int x, int y, int tIndex_size) {
    if (x == y) return false;
    for (int i = 1; i < tIndex_size - 1; i += 2) {
        if (tIndex[i] == x && tIndex[i+1] == y) return false;
        if (tIndex[i] == y && tIndex[i+1] == x) return false;
    }
    return true;
}

__device__ bool nextXPossible(int *tIndex, int i, int *tour, int tIndex_size, int *size) 
{
    return isConnected(tIndex, i, getNextIdx(i, size), tIndex_size) || isConnected(tIndex, i, getPreviousIdx(i, size), tIndex_size);
}

__device__ bool isPositiveGain(int *tIndex, int ti, int *tour, int tIndex_size, Point *d_coordinates,  long long int *edgeCount) 
{
    int gain = 0;
    for (int i = 1; i < tIndex_size - 2; ++i) {
        int t1 = tIndex[i];
        int t2 = tIndex[i+1];
        int t3 = i == tIndex_size - 3 ? ti : tIndex[i+2];

        gain += getDistance2node(t2, t3, tour, d_coordinates, edgeCount) - getDistance2node(t1, t2, tour, d_coordinates, edgeCount); // |yi| - |xi|

    }
    return gain > 0;
}

__device__ int getNextPossibleY(int *tIndex, int *tour, int tIndex_size, Point *d_coordinates, int *size, long long int *edgeCount) 
{
    int ti = tIndex[tIndex_size - 1];

    double minDistance = DBL_MAX;
    int minNode = -1;

    for (int i = 0; i < size[0]; ++i) {
        if (!isDisjunctive(tIndex, i, ti, tIndex_size)) {
            continue; // Disjunctive criteria
        }

        if (!isPositiveGain(tIndex, i, tour, tIndex_size, d_coordinates, edgeCount)) {
            continue; // Gain criteria
        };
        if (!nextXPossible(tIndex, i, tour, tIndex_size, size)) {
            continue; // Step 4.f.
        }

        // Get closest y
        if (getDistance2node(ti, i, tour, d_coordinates, edgeCount) < minDistance) 
        {
            minNode = i;
            minDistance = getDistance2node(ti, i, tour, d_coordinates, edgeCount);
        };

    }

    return minNode;
}

__device__ void  constructNewTourThree(int *tIndex, int newItem, Edge *currentEdges, int *tourOneTwo, int *tour, int tIndex_size, int *tourOT_size, int *size) 
{
    tIndex[tIndex_size] = newItem;
    tIndex_size++;
    tIndex[tIndex_size] = tIndex[1];
    tIndex_size++;

    constructNewTour_OneTwo(tIndex, currentEdges, tourOneTwo, tour, tourOT_size, tIndex_size, size);
}

__device__ int selectNewT(int *tIndex, int *tour1, Edge *currentEdges, int *tour, int tIndex_size, int *tour1_size, int *size)
{
    int option1 = getPreviousIdx(tIndex[tIndex_size - 1], size);
    int option2 = getNextIdx(tIndex[tIndex_size - 1], size);

    tour1_size[0] = 0;

    constructNewTourThree(tIndex, option1, currentEdges, tour1, tour, tIndex_size, tour1_size, size);

    if (isTour(tour1, tour1_size, size)) {
        return option1;
    }
    else {
        tour1_size[0] = 0;
        constructNewTourThree(tIndex, option2, currentEdges, tour1, tour, tIndex_size, tour1_size, size);
        if (isTour(tour1, tour1_size, size)) {
            return option2;
        }
    }
    return -1;
}

__device__ void startAlgorithm(int t1, int t2, int t3, int *tIndex, int *tour1, Edge *currentEdges, int *tour, int *tour1_size, Point *d_coordinates, int *size, long long int *edgeCount, int *improveFlag) {
    tIndex[0] = -1; // Start with the index 1 to be consistent with Lin-Kernighan Paper
    tIndex[1] = t1;
    tIndex[2] = t2;
    tIndex[3] = t3;
    double initialGain = getDistance2node(t2, t1, tour, d_coordinates, edgeCount) - getDistance2node(t3, t2, tour, d_coordinates, edgeCount); // |x1| - |y1|
    double GStar = 0;
    double Gi = initialGain;
    int k = 3;
    int tIndex_size = 4;

    for (int i = 4;; i += 2) 
    {

        int newT = selectNewT(tIndex, tour1, currentEdges, tour, tIndex_size, tour1_size, size);
        

        if (newT == -1) {
            break; // This should not happen according to the paper
        }

        tIndex[tIndex_size] = newT;
        tIndex_size++;

        int tiplus1 = getNextPossibleY(tIndex, tour, tIndex_size, d_coordinates, size, edgeCount);

        if (tiplus1 == -1) {
            break;
        }

        // Step 4.f from the paper
        Gi += getDistance2node(tIndex[tIndex_size - 2], newT, tour, d_coordinates, edgeCount);
        if (Gi - getDistance2node(newT, t1, tour, d_coordinates, edgeCount) > GStar) {
            GStar = Gi - getDistance2node(newT, t1, tour, d_coordinates, edgeCount);
            k = i;
        }

        tIndex[tIndex_size] = tiplus1;
        tIndex_size++;


        Gi -= getDistance2node(newT, tiplus1, tour, d_coordinates, edgeCount);
    }

    if (GStar > 0.001)
    {
        tIndex[k+1] = tIndex[1];
        getTPrime(tIndex, k, currentEdges, tour, size); // Update the tour

        improveFlag[0] += 1;
    }

}

__host__ __device__ int gcd(int a, int b) 
{ 
    if (b == 0) 
        return a; 
    else
        return gcd(b, a % b); 
} 

__host__ __device__ void leftRotate(int arr[], int d, int n) 
{ 
    int i, j, k, temp; 
    for (i = 0; i < gcd(d, n); i++) { 
        /* move i-th values of blocks */
        temp = arr[i]; 
        j = i; 
        while (1) { 
            k = j + d; 
            if (k >= n) 
                k = k - n; 
            if (k == i) 
                break; 
            arr[j] = arr[k]; 
            j = k; 
        } 
        arr[j] = temp; 
    } 
} 


__device__ int getNearestNeighbor(int index, int *tour, Point *d_coordinates, int *size, long long int *edgeCount)
{
    double minDistance = DBL_MAX;
    int nearestNode = -1;
    int actualNode = tour[index];

    for (int i = 0; i < size[0]; ++i) 
    {
        if (tour[i] != actualNode) 
        {
            double distance = getDistance2nodeActualNode(tour[i], actualNode, d_coordinates);
            if (distance < minDistance) 
            {
                nearestNode = i;
                minDistance = distance;
            }
        }
    }
    return nearestNode;
}

__global__ void improve_iter(int *tIndex, int *tour1, Edge *currentEdges, int *tour, int *tour1_size, Point *d_coordinates, int *size, long long int *edgeCount, int *improveFlag, int *d_dimension, int SegmentLength_v2)
{
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
    int y_ = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x_ + y_ * blockDim.x * gridDim.x;

    if(id<SegmentLength_v2)
    {
        int t1 = id;
        int t2 = getNextIdx(t1, size);
        int t3 = getNearestNeighbor(t2, tour, d_coordinates, size, edgeCount);

        if (t3 != -1 && getDistance2node(t2, t3, tour, d_coordinates, edgeCount) < getDistance2node(t1, t2, tour, d_coordinates, edgeCount)) 
            startAlgorithm(t1, t2, t3, &tIndex[id*d_dimension[0]*2], &tour1[id*d_dimension[0]], &currentEdges[id*d_dimension[0]*2], &tour[id*d_dimension[0]], &tour1_size[id], d_coordinates, size, edgeCount, improveFlag);
        else
        {
            t2 = getPreviousIdx(t1, size);
            t3 = getNearestNeighbor(t2, tour, d_coordinates, size, edgeCount);

            if (t3 != -1 && getDistance2node(t2, t3, tour, d_coordinates, edgeCount) < getDistance2node(t1, t2, tour, d_coordinates, edgeCount)) 
                startAlgorithm(t1, t2, t3, &tIndex[id*d_dimension[0]*2], &tour1[id*d_dimension[0]], &currentEdges[id*d_dimension[0]*2], &tour[id*d_dimension[0]], &tour1_size[id], d_coordinates, size, edgeCount, improveFlag);
        }       

    }
}

__global__ void checkTour(int *tour, int left, int right, int _size, int SegmentLength_v2, int *d_fullTour, int *bestList, Point *d_coordinates, int d_dimension)
{
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
    int y_ = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x_ + y_ * blockDim.x * gridDim.x;

    if(id < SegmentLength_v2)
    {
        double tmp = DBL_MAX;
        int head = 0;

        for(int i = 0; i < _size; ++i)
        {
            double dis = getDistance2nodeActualNode(tour[ (i+1)%_size + id*d_dimension ], left, d_coordinates) + getDistance2nodeActualNode(tour[i+id*d_dimension], right, d_coordinates);
            if(dis < tmp)
            {
                tmp = dis;
                head = (i+1)%_size;
            }
        }

        double sum = 0;

        for(int i = 0; i < SegmentLength_v2; i++) 
        {
            int a = tour[i + id*d_dimension];                  // <->
            int b = tour[((i+1)%SegmentLength_v2) + id*d_dimension];    // <->
            sum += getDistance2nodeActualNode(a,b,d_coordinates);
        }
    
        bestList[id] = tmp + sum;
        leftRotate(&tour[id*d_dimension], head, _size);
    }
}

__global__ void checkFullTour(int *tour, int _size, int SegmentLength_v2, int *d_fullTour, int *bestList, Point *d_coordinates, int d_dimension)
{
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
    int y_ = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x_ + y_ * blockDim.x * gridDim.x;

    if(id < SegmentLength_v2)
    {
        double sum = 0;

        for(int i = 0; i < SegmentLength_v2; i++) 
        {
            int a = tour[i + id*d_dimension];                  // <->
            int b = tour[((i+1)%SegmentLength_v2) + id*d_dimension];    // <->
            sum += getDistance2nodeActualNode(a,b,d_coordinates);
        }
    
        bestList[id] = sum;
    }
}

__global__ void printt(int *tour)
{
    for(int i=0;i<29;i++)
        {
            printf("Tour:");
        for(int j=0;j<29;j++)
            printf("%d ", tour[i*29 + j]);
printf("\n");

        }
}

__global__ void printtt(int *tour)
{

            printf("Best:");
        for(int j=0;j<29;j++)
            printf("%d ", tour[j]);
printf("\n");

        
}

void improve_one(int *tIndex, int *tour1, int *tour, Edge *currentEdges, std::vector<int> &Rotate, int *tour1_size, int *Ns, int *dimension, int *d_dimension, Point *coordinates, Point *d_coordinates, int *size, int *h_size, long long int *h_edgeCount, long long int *edgeCount, int *d_improveFlag, int *improveFlag, int *h_tour, int *bestList)
{
    int SegmentLength_v2 = SegmentLength;
    int flag = 0;

    if(dimension[0] % SegmentLength!=0 && dimension[0] % SegmentLength > 4)
    {
        flag = dimension[0] % SegmentLength;
        Ns[0] = Ns[0] + 1;
    }

    for(int seg = 0; seg < Ns[0]; seg++)
    {
        double old_distance = 0;
        if(flag!=0 && seg==Ns[0]-1)
            SegmentLength_v2 = flag;


        improveFlag[0] = 0;
        cudaMemcpy       (d_improveFlag, improveFlag, 1 * sizeof(int), cudaMemcpyHostToDevice);

        old_distance = getDistance(dimension);
        cudaMemcpy       (h_edgeCount, edgeCount, 1 * sizeof(long long int), cudaMemcpyDeviceToHost);
        //printf("%lld, %lf %d\n", h_edgeCount[0], old_distance, SegmentLength);
        fprintf(fp,"%lf,", old_distance);
        fprintf(ws_fp,"%lld,", h_edgeCount[0]);
        fprintf(runtime_fp,"%lf,", ( clock() - sub_start ) / (double) CLOCKS_PER_SEC);

        printf("%lf\n", old_distance);

        fflush(fp);
        fflush(ws_fp);
        fflush(runtime_fp);

        for(int k=0;k<dimension[0];k++)
            tmp_fullTour[k] = fullTour[k];

        int sbutour_size = 0;
        

        if(SegmentLength_v2 == dimension[0])
        {
            for(int k=0;k<SegmentLength_v2;k++,sbutour_size++)
                h_tour[sbutour_size] = fullTour[seg*SegmentLength + k];
            h_size[0] = SegmentLength_v2;

            cudaMemcpy       (size, h_size, 1 * sizeof(int), cudaMemcpyHostToDevice);
            for(int core = 0; core < CORE; core++)
                cudaMemcpy       (&tour[core*dimension[0]], h_tour, dimension[0] * sizeof(int), cudaMemcpyHostToDevice);

            improve_iter <<<iDivUp(CORE, 32), 32>>> (tIndex, tour1, currentEdges, tour, tour1_size, d_coordinates, size, edgeCount, d_improveFlag, d_dimension, SegmentLength_v2);
                //cudaDeviceSynchronize();
        }
        else
        {
            for(int k=1;k<SegmentLength_v2-1;k++,sbutour_size++)
                h_tour[sbutour_size] = fullTour[seg*SegmentLength + k];
            h_size[0] = SegmentLength_v2-2;
            cudaMemcpy       (size, h_size, 1 * sizeof(int), cudaMemcpyHostToDevice);

            for(int core = 0; core < CORE; core++)
                cudaMemcpy       (&tour[core*dimension[0]], h_tour, dimension[0] * sizeof(int), cudaMemcpyHostToDevice);

            improve_iter <<<CORE, 1>>> (tIndex, tour1, currentEdges, tour, tour1_size, d_coordinates, size, edgeCount, d_improveFlag, d_dimension, SegmentLength_v2);
        }

        double new_distance;

        cudaMemcpy       (improveFlag, d_improveFlag, 1 * sizeof(int), cudaMemcpyDeviceToHost);

        if(improveFlag[0]==0)
            NoImprovedNumber += 1;
        else
        {   
            int iter = SegmentLength_v2;
            if(SegmentLength_v2!=dimension[0])
            {
                int left  = fullTour[seg*SegmentLength];
                int right = fullTour[seg*SegmentLength + SegmentLength_v2 -1];
    
                cudaMemcpy       (tIndex, fullTour, sizeof(int) * dimension[0], cudaMemcpyHostToDevice);
                checkTour <<<dimension[0],1>>> (tour, left, right, h_size[0], SegmentLength_v2, tIndex, tour1, d_coordinates, dimension[0]);
                cudaMemcpy       (bestList, tour1, sizeof(int) * dimension[0], cudaMemcpyDeviceToHost);
                int tmp = INT_MAX;
                int best_id;
                for(int i=0;i<SegmentLength_v2;i++)
                    if(bestList[i] < tmp)
                    {
                        tmp = bestList[i];
                        best_id = i;
                    }
    
                cudaMemcpy       (h_tour, &tour[best_id*dimension[0]] , dimension[0] * sizeof(int), cudaMemcpyDeviceToHost);
                iter = SegmentLength_v2-2;
                for(int i = 0; i < iter; ++i)
                    fullTour[seg*SegmentLength + 1 + i] = h_tour[i];

                new_distance = getDistance(dimension);
                
                if(old_distance <= new_distance)
                {
                    for(int k=0;k<dimension[0];k++)
                        fullTour[k] = tmp_fullTour[k];
                    NoImprovedNumber += 1;

                }
            }

            else
            {
                cudaMemcpy       (tIndex, fullTour, sizeof(int) * dimension[0], cudaMemcpyHostToDevice);
    
                checkFullTour <<<dimension[0],1>>> (tour, h_size[0], SegmentLength_v2, tIndex, tour1, d_coordinates, dimension[0]);
                cudaMemcpy       (bestList, tour1, sizeof(int) * dimension[0], cudaMemcpyDeviceToHost);

                int tmp = INT_MAX;
                int best_id;
                for(int i=0;i<SegmentLength_v2;i++)
                    if(bestList[i] < tmp)
                    {
                        tmp = bestList[i];
                        best_id = i;
                    }
                cudaMemcpy       (h_tour, &tour[best_id*dimension[0]] , dimension[0] * sizeof(int), cudaMemcpyDeviceToHost);

                for(int i = 0; i < SegmentLength_v2; ++i)
                    fullTour[seg*SegmentLength + i] = h_tour[i];
            }
                
        }
    }
}

void runAlgorithm(int *tIndex, int *tour1, int *tour, Edge *currentEdges, std::vector<int> &Rotate, int *tour1_size, int *Ns, int *dimension, int *d_dimension, Point *coordinates, Point *d_coordinates, int *size, int *h_size, long long int *h_edgeCount, long long int *edgeCount, int *d_improveFlag, int *improveFlag, int *h_tour, int *bestList) 
{
    double oldDistance = 0;
    double newDistance = getDistance(dimension);

    cudaMemset(edgeCount, 0, sizeof(long long int) * 1);
    while(true)
    {
        oldDistance = newDistance;
        NoImprovedNumber = 0;
        Ns[0] = dimension[0] / SegmentLength;

        if(SegmentLength!=dimension[0])
        {
            Rotate.clear();
            for(int f=0;f<dimension[0];f++)
                Rotate.push_back(fullTour[f]);
            int offset = rand()%dimension[0];
            std::rotate(Rotate.begin(), Rotate.begin()+offset, Rotate.end());
            for(int f=0;f<dimension[0];f++)
                fullTour[f] = Rotate[f];
        }
    
        improve_one (tIndex, tour1, tour, currentEdges, Rotate, tour1_size, Ns, dimension, d_dimension, coordinates, d_coordinates, size, h_size, h_edgeCount, edgeCount, d_improveFlag, improveFlag, h_tour, bestList);
 
        newDistance = getDistance(dimension);
        if(newDistance >= oldDistance && SegmentLength==dimension[0])
            break;

        if( ((double)NoImprovedNumber/(double)Ns[0]) > 0.1 )
            SegmentLength *= 2;

        if(SegmentLength > dimension[0])
            SegmentLength = dimension[0];
        
    }

}


int main(int argc, char *argv[])
{
    clock_t start;
    double duration = 0;
    int TRAILS = 10;
    int init_seglen;
    srand(time(NULL));


    sleep(rand()%10);
    //
    //srand(1);

    int *dimension;
    int *d_dimension;
    dimension      = (int*) malloc( 1 * sizeof(int));
    cudaMalloc ( (void **) &d_dimension, sizeof (int) * 1);
    dimension[0] = countcity(argv[1]);
    cudaMemcpy       (d_dimension, dimension , 1 * sizeof(int),cudaMemcpyHostToDevice);  

    init_seglen = atoi(argv[2]);
    power  = atof(argv[3]);    
    sita   = atof(argv[4]);
    DEVICE_ID  = atoi(argv[5]);
    int serial = atoi(argv[6]);

    cudaSetDevice(DEVICE_ID);

    Point *coordinates;
    Point *d_coordinates;

    cudaMalloc ( (void **) &d_coordinates,   sizeof (Point)  * dimension[0]);
    coordinates = (Point*) malloc( dimension[0] * sizeof(Point));
    readcity_coords(argv[1], coordinates);
    cudaMemcpy       (d_coordinates, coordinates , dimension[0] * sizeof(Point),cudaMemcpyHostToDevice);  
    initDistanceTable(dimension, coordinates);

    
    struct stat st = {0};
    char filename[512];
    char ws_filename[512];
    char runtime_filename[512];

    char output_file[100]="./output/";
    for(int i=1;i<100;i++) if(argv[1][i]=='.') argv[1][i]='\0';
    strcat (output_file,argv[1]);
    if (stat(output_file, &st) == -1)
        mkdir(output_file, 0777);

    sprintf(filename,"./output/%s/cuda_sLKH_N%d_InitSeg%d_power%f_sita%f_serial%d.csv",argv[1],dimension[0], init_seglen, power, sita, serial);
    sprintf(ws_filename,"./output/%s/WS_cuda_sLKH_N%d_InitSeg%d_power%f_sita%f_serial%d.csv",argv[1],dimension[0], init_seglen, power, sita, serial);
    sprintf(runtime_filename,"./output/%s/Runtime_cuda_sLKH_N%d_InitSeg%d_power%f_sita%f_serial%d.csv",argv[1],dimension[0], init_seglen, power, sita, serial);

    ws_fp=fopen(ws_filename,"w+");
    fp=fopen(filename,"w+");
    runtime_fp=fopen(runtime_filename,"w+");


    std::vector <int> Rotate(dimension[0]);
    
    fullTour             = (int*) malloc( dimension[0] * sizeof(int));
    tmp_fullTour         = (int*) malloc( dimension[0] * sizeof(int));

    long long int *edgeCount;
    long long int *h_edgeCount;
    
    int *size;
    int *h_size;
    int *improveFlag, *d_improveFlag;
    int *Ns;
    
    

    cudaMalloc ( (void **) &edgeCount,     sizeof (long long int)  * 1);    //
    cudaMalloc ( (void **) &size,          sizeof (int)  * 1);
    cudaMalloc ( (void **) &d_improveFlag, sizeof (int)  * CORE);            //
    
    h_edgeCount = (long long int*) malloc( 1 * sizeof(long long int));
    improveFlag = (int*) malloc( 1 * sizeof(int));
    h_size = (int*) malloc( 1 * sizeof(int));
    Ns = (int*) malloc( 1 * sizeof(int));


    int *tour1_size;
    int *bestList = (int*) malloc( dimension[0] * sizeof(int));


    int *tour;
    int *h_tour;
    h_tour = (int*) malloc( dimension[0] * sizeof(int));
    int *tour1;
    int *tIndex;
    Edge *currentEdges;
    gpuErrchk( cudaMalloc ( (void **) &tour1_size,   sizeof (int)  * CORE));
    gpuErrchk( cudaMalloc ( (void **) &tour,         sizeof (int)  * dimension[0] * CORE));
    gpuErrchk( cudaMalloc ( (void **) &tour1,        sizeof (int)  * dimension[0] * CORE));
    gpuErrchk( cudaMalloc ( (void **) &tIndex,       sizeof (int)  * dimension[0] * CORE * 2));
    gpuErrchk( cudaMalloc ( (void **) &currentEdges, sizeof (Edge) * dimension[0] * CORE * 2));



    while(TRAILS--)
    {
        start = clock();

        
        SegmentLength = init_seglen;

        createRandomTour(dimension);

        sub_start = clock();
        runAlgorithm(tIndex, tour1, tour, currentEdges, Rotate, tour1_size, Ns, dimension, d_dimension, coordinates, d_coordinates, size, h_size, h_edgeCount, edgeCount, d_improveFlag, improveFlag, h_tour, bestList);
        
        duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;

        std::cout<<"Times: "<< duration << " RUNS" << TRAILS;
        printf("cuda_sLKH_N%d_InitSeg%d_power%f_sita%f\n", dimension[0], init_seglen, power, sita);
        fprintf(fp,"\n");
        fprintf(ws_fp,"\n");
        fprintf(runtime_fp,"\n");
        fflush(stdout);
    }



}
