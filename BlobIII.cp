#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

void grayScale(const Mat &source, Mat &destination){
    int g;
    if (destination.empty())
        destination = Mat(source.rows, source.cols, source.type());

    for(int i = 0; i<source.rows; i++){
        for(int j = 0; j<source.cols;j++){
            /*
            g = source.at<Vec3b>(i, source.cols-1-j)[0] + source.at<Vec3b>(i, source.cols-1-j)[1] + source.at<Vec3b>(i, source.cols-1-j)[2];
            g = g/3;
            destination.at<Vec3b>(i,source.cols-1-j)[0] = g;
            destination.at<Vec3b>(i,source.cols-1-j)[1] = g;
            destination.at<Vec3b>(i,source.cols-1-j)[2] = g;
            */    
            g = source.at<Vec3b>(i,j)[0] + source.at<Vec3b>(i,j)[1] + source.at<Vec3b>(i,j)[2];
            g = g/3;
            destination.at<Vec3b>(i,j)[0] = g;
            destination.at<Vec3b>(i,j)[1] = g;
            destination.at<Vec3b>(i,j)[2] = g;
        }
    }
}

void binaryFilter(const Mat &source, Mat &destination, int threshold){
    int g;
    if (destination.empty())
        destination = Mat(source.rows, source.cols, source.type());

    for(int i = 0; i< source.rows; i++){
        for(int j = 0; j< source.cols;j++){
            /*
            if(source.at<Vec3b>(i, source.cols-1-j)[0] <= threshold)
                g = 0;
            else
                g = 255;
            destination.at<Vec3b>(i,source.cols-1-j)[0] = g;
            destination.at<Vec3b>(i,source.cols-1-j)[1] = g;
            destination.at<Vec3b>(i,source.cols-1-j)[2] = g;
            */    
            if(source.at<Vec3b>(i,j)[0] <= threshold)
                g = 0;
            else
                g = 255;
            destination.at<Vec3b>(i,j)[0] = g;
            destination.at<Vec3b>(i,j)[1] = g;
            destination.at<Vec3b>(i,j)[2] = g;
        }
    }
}

void blobColouring(const Mat &source, Mat &destination, Mat &regions){
    int colors[10][3];
    colors[0][0] = 255; colors[0][1] = 255; colors[0][2] = 255;
    colors[1][0] =   0; colors[1][1] = 255; colors[1][2] =   0;
    colors[2][0] =   0; colors[2][1] =   0; colors[2][2] = 255;
    colors[3][0] = 255; colors[3][1] = 255; colors[3][2] =   0;
    colors[4][0] = 255; colors[4][1] =   0; colors[4][2] = 255;
    colors[5][0] =   0; colors[5][1] = 255; colors[5][2] = 255;
    colors[6][0] =   0; colors[6][1] =   0; colors[6][2] =   0;
    colors[7][0] = 255; colors[7][1] =   0; colors[7][2] =   0;

    string colorName[] = {"White","Green","Red","Cyan","Magenta","Yellow","Black","Blue"};

    cout<<"Rows: "<<source.rows<<" Columns: "<<source.cols<<endl;

    int k = 0;
    int lookUpTable[1000][4];           //Column 0 contains the region index
                                        //Column 1 contains the areas
                                        //Column 2 contains ∑ X
                                        //Column 3 contains ∑ Y
                                        //Column 4 contains ∑ X^2
                                        //Column 5 contains ∑ Y^2
                                        //Column 6 contains ∑ X * Y

    for(int i = 0; i<1000; i++) {
        lookUpTable[i][0] = 0;
        lookUpTable[i][1] = 0;
        lookUpTable[i][2] = 0;
        lookUpTable[i][3] = 0;
        lookUpTable[i][4] = 0;
        lookUpTable[i][5] = 0;
        lookUpTable[i][6] = 0;
    }

    lookUpTable[0][0] = 0;
    int temporal[source.rows][source.cols];

    if (destination.empty()){
        destination = Mat(source.rows, source.cols, source.type());
        regions = Mat(source.rows, source.cols, source.type());
    }
    
    for(int i = 0; i< source.rows; i++){
        for(int j = 0; j< source.cols;j++){
            temporal[i][j] = 0;
        }
    }
    
    for(int i = 1; i< source.rows; i++){
        for(int j = 1; j< source.cols;j++){
            if(source.at<Vec3b>(i,j)[0] == 0){
                //cout<<k<<" "<<temporal[i-1][j]<<" "<<temporal[i][j-1]<<endl;
                if(source.at<Vec3b>(i-1,j)[0] == 255 && source.at<Vec3b>(i,j-1)[0] == 255){
                    k = k+1;
                    temporal[i][j] = k;
                    lookUpTable[k][0] = k;
                    lookUpTable[k][1]++;
                    lookUpTable[k][2] = j;
                    lookUpTable[k][3] = i;
                    lookUpTable[k][4] = j^2;
                    lookUpTable[k][5] = i^2;
                    lookUpTable[k][6] = j*i;
                    destination.at<Vec3b>(i,j)[0] = colors[k%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[k%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[k%7+1][2];
                }
                else if(source.at<Vec3b>(i-1,j)[0] == 255 && source.at<Vec3b>(i,j-1)[0] == 0){
                    int left = lookUpTable[temporal[i][j-1]][0];
                    //lookUpTable[temporal[i][j]][0] = left;
                    temporal[i][j] = temporal[i][j-1];
                    lookUpTable[left][1]++;
                    lookUpTable[left][2]+= j;
                    lookUpTable[left][3]+= i;
                    lookUpTable[left][4]+= j^2;
                    lookUpTable[left][5]+= i^2;
                    lookUpTable[left][6]+= j*i;
                    //lookUpTable[k][0] = temporal[i][j-1];
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i][j-1])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i][j-1])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i][j-1])%7+1][2];
                }
                else if(source.at<Vec3b>(i-1,j)[0] == 0 && source.at<Vec3b>(i,j-1)[0] == 255){
                    int top = lookUpTable[temporal[i-1][j]][0];
                    //lookUpTable[temporal[i][j]][0] = left;
                    temporal[i][j] = temporal[i-1][j];
                    lookUpTable[top][1]++;
                    lookUpTable[top][2]+= j;
                    lookUpTable[top][3]+= i;
                    lookUpTable[top][4]+= j^2;
                    lookUpTable[top][5]+= i^2;
                    lookUpTable[top][6]+= j*i;
                    //lookUpTable[k][0] = temporal[i-1][j];
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i-1][j])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i-1][j])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i-1][j])%7+1][2];
                }
                
                else if(source.at<Vec3b>(i-1,j)[0] == 0 && source.at<Vec3b>(i,j-1)[0] == 0){
                    int left, top;
                    top = lookUpTable[temporal[i-1][j]][0]; 
                    left = lookUpTable[temporal[i][j-1]][0];              
                    if(top <= left ){
                        //lookUpTable[temporal[i][j]][0] = top;
                        temporal[i][j]= temporal[i-1][j];
                        lookUpTable[top][1]++;
                        lookUpTable[top][2]+= j;
                        lookUpTable[top][3]+= i;
                        lookUpTable[top][4]+= j^2;
                        lookUpTable[top][5]+= i^2;
                        lookUpTable[top][6]+= j*i;
                        //lookUpTable[k][0] = temporal[i-1][j];
                        lookUpTable[left][0] = top;
                    }
                    else{
                        //lookUpTable[temporal[i][j]][0] = left;
                        temporal[i][j] = temporal[i][j-1];
                        lookUpTable[left][1]++;
                        lookUpTable[left][2]+= j;
                        lookUpTable[left][3]+= i;
                        lookUpTable[left][4]+= j^2;
                        lookUpTable[left][5]+= i^2;
                        lookUpTable[left][6]+= j*i;
                        //lookUpTable[k][0] = temporal[i][j-1];
                        lookUpTable[top][0] = left;
                    }
                    
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i][j])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i][j])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i][j])%7+1][2];
                } 
                            
                /*
                else if(temporal[i-1][j] > 0 && temporal[i][j-1] > 0){
                    temporal[i][j] = temporal[i][j-1];
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i][j-1])%10][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i][j-1])%10][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i][j-1])%10][2];
                }
                */
            }
            else{
                destination.at<Vec3b>(i,j)[0] = 255;
                destination.at<Vec3b>(i,j)[1] = 255;
                destination.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
    
    /*
    for(int i = 0; i<200; i++)
        cout<<i<<": "<<lookUpTable[i][0]<<endl;
    */
    
    int finalIndex, tempArea, tempX, tempY, tempXSquare, tempYSquare, tempXY;
    for(int i = 0; i< source.rows; i++){
        for(int j = 0; j< source.cols;j++){
            if(temporal[i][j] > 0){
                tempArea = 0;
                finalIndex = lookUpTable[temporal[i][j]][0];
                while(lookUpTable[finalIndex][0] != finalIndex){
                    tempArea = lookUpTable[finalIndex][1];
                    tempX    = lookUpTable[finalIndex][2];
                    tempY    = lookUpTable[finalIndex][3];
                    tempXSquare    = lookUpTable[finalIndex][4];
                    tempYSquare    = lookUpTable[finalIndex][5];
                    tempXY   = lookUpTable[finalIndex][6];  
                    lookUpTable[finalIndex][1] = 0;
                    lookUpTable[finalIndex][2] = 0;
                    lookUpTable[finalIndex][3] = 0;
                    lookUpTable[finalIndex][4] = 0;
                    lookUpTable[finalIndex][5] = 0;
                    lookUpTable[finalIndex][6] = 0;
                    finalIndex = lookUpTable[finalIndex][0];
                    lookUpTable[finalIndex][1] += tempArea;
                    lookUpTable[finalIndex][2] += tempX;
                    lookUpTable[finalIndex][3] += tempY;
                    lookUpTable[finalIndex][4] += tempXSquare;
                    lookUpTable[finalIndex][5] += tempYSquare;
                    lookUpTable[finalIndex][6] += tempXY;
                    //cout<<"X Sum: "<<tempX<<endl;
                }
                regions.at<Vec3b>(i,j)[0] = colors[lookUpTable[finalIndex][0]%7+1][0];
                regions.at<Vec3b>(i,j)[1] = colors[lookUpTable[finalIndex][0]%7+1][1];
                regions.at<Vec3b>(i,j)[2] = colors[lookUpTable[finalIndex][0]%7+1][2];
            }
            else{
                regions.at<Vec3b>(i,j)[0] = colors[0][0];
                regions.at<Vec3b>(i,j)[1] = colors[0][1];
                regions.at<Vec3b>(i,j)[2] = colors[0][2];
            }
        }
    }
    for(int i=0; i<k;i++){
        //cout<<i<<" "<<lookUpTable[i][0]<<endl;
        if(lookUpTable[i][1] > 200){
            printf("Region: %3d Area: %6d pixels  Color: %7s  Xc: %4d  Yc: %4d\n", 
                    i,lookUpTable[i][1],colorName[(i%7)+1].c_str(),(lookUpTable[i][2]/lookUpTable[i][1]),(lookUpTable[i][3]/lookUpTable[i][1]));
            //printf("Region: %3d K: %3d Area: %6d pixels %s\n", i,lookUpTable[i][0],lookUpTable[i][1],colorName[(i%7)+1].c_str());
            //cout<<"Region "<<i<<" Area: "<<lookUpTable[i][1]<<" pixels Color: "<<colorName[(i%7)+1]<<endl;
            line( 
                regions, 
                Point( lookUpTable[i][2]/lookUpTable[i][1], lookUpTable[i][3]/lookUpTable[i][1]),
                Point( lookUpTable[i][2]/lookUpTable[i][1], lookUpTable[i][3]/lookUpTable[i][1]),
                Scalar(0,0,0),
                5, 
                8, 
                0  
                );
            //Scalar(colors[(lookUpTable[finalIndex][0]+1)%7+1][0], colors[(lookUpTable[finalIndex][0]+1)%7+1][1], colors[(lookUpTable[finalIndex][0]+1)%7+1][2]), 
        }
    }
    cout<<endl;   
}


int main(int argc, char** argv )
{
    Mat image, resized, gray, binaryF,blobDetection,colouredRegions;                 
    int threshold = 240;
 /*
    VideoCapture cap;

    if(!cap.open(0))  
        return -1;

    while(1){
        Mat frame;
        cap >> frame; 
        //image = imread( "SegmentationFigures2.jpg", 1);
        cv::resize(frame, resized, cv::Size(), 0.6,0.6);
        grayScale(resized,gray);
        binaryFilter(gray,binaryF,threshold);       
        //imshow("Test", gray);
        //cv::imshow("Test", binaryF);
        //waitKey(0);
        blobColouring(binaryF,blobDetection,colouredRegions);
        imshow("Blob Colouring"   , blobDetection);
        imshow("Coloured Regions ", colouredRegions);

        if(waitKey(30) >= 0){
            waitKey(0);
            break;
        }
    }
*/
    image = imread( "SegmentationFigures2.jpg", 1);
    cv::resize(image, resized, cv::Size(), 0.6,0.6);
    grayScale(resized,gray);
    binaryFilter(gray,binaryF,threshold);       
    //imshow("Test", gray);
    //cv::imshow("Test", binaryF);
    blobColouring(binaryF,blobDetection,colouredRegions);
    imshow("Blob Colouring"   , blobDetection);
    imshow("Coloured Regions ", colouredRegions);

    waitKey(0);

    return 0;
}

/**

m_00 = lookUpTable[k][1]
m_10 = lookUpTable[k][2];
m_01 = lookUpTable[k][3];
m_20 = lookUpTable[k][4];
m_02 = lookUpTable[k][5];
m_11 = lookUpTable[k][6];

u_20 = m_20 - m_10 * m_10/m_00;
u_02 = m_20 - m_01 * m_01/m_00;

**/
