#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
/*
 * A simple 'getting started' interface to the ARDrone, v0.2 
 * author: Tom Krajnik
 * The code is straightforward,
 * check out the CHeli class and main() to see 
 */
#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

bool stop = false;
CRawImage *image;
CHeli *heli;
float pitch, roll, yaw, height;
int hover;

// Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
int joypadRoll, joypadPitch, joypadVerticalSpeed, joypadYaw;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
string ultimo = "init";

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

//Fill Values for HSV Filtering
int hMin = 0;
int hMax = 0;
int sMin = 0;
int sMax = 0; 
int vMin = 184;
int vMax = 248;

char meanMat3[3][3]  =  {{1,1,1}, 
                         {1,1,1},
                         {1,1,1}};

void imageHSVFunction(const Mat &sourceImage, Mat &FilteredImage){
    int max,min, v, h,r,g,b;
    double s;

    if (FilteredImage.empty())
        FilteredImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

    for (int t = 0; t < sourceImage.rows; ++t)
        for (int x = 0; x < sourceImage.cols / 2; ++x){
            //First Half
            r = sourceImage.at<Vec3b>(t, x)[2];
            g = sourceImage.at<Vec3b>(t, x)[1];
            b = sourceImage.at<Vec3b>(t, x)[0];
            max = 0;
            if(r>=g && r>=b)    max = r;
            if(g>=b && g>=r)    max = g;
            if(b>=r && b>=g)    max = b;
            v = max;

            min = 0;
            if(r<g && r<b)  min = r;
            if(g<b && g<r)  min = g;
            if(b<r && b<g)  min = b;
            
            if (v==min)
                v = v+1;
        
            if(v != 0)
                s = (v-min)/v;
            else
                s = 0;
        
            if(v == r)
                h = 60*(((g-b)/(v-min))%6);
            if(v == g)
                h = 120 + 60*(b-r)/(v-min);
            if(v == b)
                h = 240 + 60*(r-g)/(v-min);
            if(h< 0)
                h = h + 360;
        
            h = (h*255)/360;
            s *= 255;

            FilteredImage.at<Vec3b>(t, x)[0] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
            FilteredImage.at<Vec3b>(t, x)[1] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
            FilteredImage.at<Vec3b>(t, x)[2] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
        
            //Second Half      
            r = sourceImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[2];
            g = sourceImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[1];
            b = sourceImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[0];
            
            max = 0;

            if(r>=g && r>=b)    max = r;
            if(g>=b && g>=r)    max = g;
            if(b>=r && b>=g)    max = b;

            v = max;

            min = 0;
            if(r<g && r<b)  min = r;
            if(g<b && g<r)  min = g;
            if(b<r && b<g)  min = b;
            
            if (v==min)
                v = v+1;
        
            if(v != 0)
                s = (v-min)/v;
            else
                s = 0;
        
            if(v == r)
                h = 60*(((g-b)/(v-min))%6);
            if(v == g)
                h = 120 + 60*(b-r)/(v-min);
            if(v == b)
                h = 240 + 60*(r-g)/(v-min);
            if(h< 0)
                h = h + 360;

            h = (h*255)/360;
            s *= 255;

            FilteredImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[0] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
            FilteredImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[1] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
            FilteredImage.at<Vec3b>(t, sourceImage.cols - 1 - x)[2] = (h <= hMax && h >= hMin && s <= sMax && s >= sMin && v <= vMax && v >= vMin)? 0: 255;
        }
}

void dilateFilter(Mat &source, Mat &destination){
    int val,size;
    bool on;
    size = 3;
    if (destination.empty())
        destination = Mat(source.rows, source.cols, source.type());

    for(int i = 0; i< source.rows; i++){
        for(int j = 0; j< source.cols;j++){
            if(i>0 && i<source.rows-(size/2) && j>0 && j<source.cols-(size/2)){
                on = false;
                for(int k = 0; k < size && !on; k++){
                    for(int l = 0; l < size && !on; l++){
                        val = meanMat3[k][l]*source.at<Vec3b>(i-(size/2)+k, j-(size/2)+l)[0];
                        if(val < 255)
                            on = true;
                    }   
                }
                if(on)
                    val = 0;
                else
                    val = 255;
            }
            else
                val = source.at<Vec3b>(i,j)[0];

            destination.at<Vec3b>(i,j)[0] = val;
            destination.at<Vec3b>(i,j)[1] = val;
            destination.at<Vec3b>(i,j)[2] = val;
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
    double lookUpTable[1000][7];        //Column 0 contains the region index
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
                if(source.at<Vec3b>(i-1,j)[0] == 255 && source.at<Vec3b>(i,j-1)[0] == 255){
                    k = k+1;
                    temporal[i][j] = k;
                    lookUpTable[k][0] = k;
                    lookUpTable[k][1]++;
                    lookUpTable[k][2] = j;
                    lookUpTable[k][3] = i;
                    lookUpTable[k][4] = (j*j);
                    lookUpTable[k][5] = (i*i);
                    lookUpTable[k][6] = (j*i);
                    destination.at<Vec3b>(i,j)[0] = colors[k%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[k%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[k%7+1][2];
                }
                else if(source.at<Vec3b>(i-1,j)[0] == 255 && source.at<Vec3b>(i,j-1)[0] == 0){
                    int left = lookUpTable[temporal[i][j-1]][0];
                    temporal[i][j] = temporal[i][j-1];
                    lookUpTable[left][1]++;
                    lookUpTable[left][2]+= j;
                    lookUpTable[left][3]+= i;
                    lookUpTable[left][4] = lookUpTable[left][4] + (j*j);
                    lookUpTable[left][5] = lookUpTable[left][5] + (i*i);
                    lookUpTable[left][6] = lookUpTable[left][6] + (j*i);
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i][j-1])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i][j-1])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i][j-1])%7+1][2];
                }
                else if(source.at<Vec3b>(i-1,j)[0] == 0 && source.at<Vec3b>(i,j-1)[0] == 255){
                    int top = lookUpTable[temporal[i-1][j]][0];
                    temporal[i][j] = temporal[i-1][j];
                    lookUpTable[top][1]++;
                    lookUpTable[top][2]+= j;
                    lookUpTable[top][3]+= i;
                    lookUpTable[top][4] = lookUpTable[top][4] + (j*j);
                    lookUpTable[top][5] = lookUpTable[top][5] + (i*i);
                    lookUpTable[top][6] = lookUpTable[top][6] + (j*i);
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i-1][j])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i-1][j])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i-1][j])%7+1][2];
                }
                
                else if(source.at<Vec3b>(i-1,j)[0] == 0 && source.at<Vec3b>(i,j-1)[0] == 0){
                    int left, top;
                    top = lookUpTable[temporal[i-1][j]][0]; 
                    left = lookUpTable[temporal[i][j-1]][0];              
                    if(top <= left ){
                        temporal[i][j]= temporal[i-1][j];
                        lookUpTable[top][1]++;
                        lookUpTable[top][2]+= j;
                        lookUpTable[top][3]+= i;
                        lookUpTable[top][4] = lookUpTable[top][4] + (j*j);
                        lookUpTable[top][5] = lookUpTable[top][5] + (i*i);
                        lookUpTable[top][6] = lookUpTable[top][6] + (j*i);
                        lookUpTable[left][0] = top;
                    }
                    else{
                        temporal[i][j] = temporal[i][j-1];
                        lookUpTable[left][1]++;
                        lookUpTable[left][2]+= j;
                        lookUpTable[left][3]+= i;
                        lookUpTable[left][4] = lookUpTable[left][4] + (j*j);
                        lookUpTable[left][5] = lookUpTable[left][5] + (i*i);
                        lookUpTable[left][6] = lookUpTable[left][6] + (j*i);
                        lookUpTable[top][0] = left;
                    }
                    
                    destination.at<Vec3b>(i,j)[0] = colors[(temporal[i][j])%7+1][0];
                    destination.at<Vec3b>(i,j)[1] = colors[(temporal[i][j])%7+1][1];
                    destination.at<Vec3b>(i,j)[2] = colors[(temporal[i][j])%7+1][2];
                } 
            }
            else{
                destination.at<Vec3b>(i,j)[0] = 255;
                destination.at<Vec3b>(i,j)[1] = 255;
                destination.at<Vec3b>(i,j)[2] = 255;
            }
        }
    }
    
    int finalIndex, tempArea, tempX, tempY, tempXSquare, tempYSquare, tempXY;
    for(int i = 0; i< source.rows; i++){
        for(int j = 0; j< source.cols;j++){
            if(temporal[i][j] > 0){
                tempArea = 0;
                finalIndex = lookUpTable[temporal[i][j]][0];
                while(lookUpTable[finalIndex][0] != finalIndex){
                    tempArea        = lookUpTable[finalIndex][1];
                    tempX           = lookUpTable[finalIndex][2];
                    tempY           = lookUpTable[finalIndex][3];
                    tempXSquare     = lookUpTable[finalIndex][4];
                    tempYSquare     = lookUpTable[finalIndex][5];
                    tempXY          = lookUpTable[finalIndex][6];  
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
                }
                regions.at<Vec3b>(i,j)[0] = colors[int(lookUpTable[finalIndex][0])%7+1][0];
                regions.at<Vec3b>(i,j)[1] = colors[int(lookUpTable[finalIndex][0])%7+1][1];
                regions.at<Vec3b>(i,j)[2] = colors[int(lookUpTable[finalIndex][0])%7+1][2];
            }
            else{
                regions.at<Vec3b>(i,j)[0] = colors[0][0];
                regions.at<Vec3b>(i,j)[1] = colors[0][1];
                regions.at<Vec3b>(i,j)[2] = colors[0][2];
            }
        }
    }
    double m_00, m_10, m_01, m_20, m_02, m_11, u_20, u_02, u_11, n_20, n_02, n_11, phi_1, phi_2, theta;
    for(int i=0; i<k;i++){
        if(lookUpTable[i][1] > 200){
            // Momentos geometricos de orden p y q m_pq = ∑_x∑_y   x^p * y^q * f(x,y):
            // con f(x,y) = 1;
            
            m_00 = lookUpTable[i][1];
            m_10 = lookUpTable[i][2];
            m_01 = lookUpTable[i][3];
            m_20 = lookUpTable[i][4];
            m_02 = lookUpTable[i][5];
            m_11 = lookUpTable[i][6];
    
            u_20 = m_20 - m_10 * (m_10/m_00);
            u_02 = m_02 - m_01 * (m_01/m_00);
            u_11 = m_11 - (m_01/m_00) * m_10;

            //cout<<m_20-m_02<<" "<<2*m_11<<endl;
    
            //Momentos centrales normalizados de 2ndo orden:
            n_20 = (m_20 - m_10 * (m_10 / m_00)) / (m_00 * m_00);
            n_02 = (m_02 - m_01 * (m_01 / m_00)) / (m_00 * m_00);
            n_11 = (m_11 - m_10 * (m_01 / m_00)) / (m_00 * m_00);
    
            //Momentos de Hu
            phi_1 = n_20 + n_02;
            phi_2 = pow(n_20 - n_02, 2) + 4 * n_11 * n_11;
            theta = (1.0/2.0) * atan2(2 * u_11, u_20 - u_02);
            //cout<<theta<<endl;
            //phi : ángulo de la recta entre centroides
            //theta : ángulo de orientación del robot

            //cout<<2 * u_11<<" "<<u_20 - u_02<<endl;
            
            //if(theta < 0)
            //    theta = theta + 6.2831853;
            double angle = theta*180.0/3.14159265;

            int a = 15;
            int b = a;
            while(a*tan(theta) > 15 || a*tan(theta) < -15)
                a = a - 0.2;

            double xC = lookUpTable[i][2]/lookUpTable[i][1];
            double yC = lookUpTable[i][3]/lookUpTable[i][1];
            printf("Region: %3d Area: %6d pixels  Color: %7s  Xc: %4d  Yc: %4d Theta: %7.3f Phi1: %8.6f Phi2: %8.6f\n", 
                    i,int(lookUpTable[i][1]),colorName[(i%7)+1].c_str(),int(lookUpTable[i][2]/lookUpTable[i][1]),
                    int(lookUpTable[i][3]/lookUpTable[i][1]),angle, phi_1, phi_2);
            line( 
                regions, 
                Point( lookUpTable[i][2]/lookUpTable[i][1], lookUpTable[i][3]/lookUpTable[i][1]),
                Point( lookUpTable[i][2]/lookUpTable[i][1], lookUpTable[i][3]/lookUpTable[i][1]),
                Scalar(0,0,0),
                4, 
                8, 
                0  
                );

            if(abs(a) < 1){
                line( 
                    regions, 
                    Point(xC, yC+b),
                    Point(xC, yC-b),
                    Scalar(0,0,0),
                    1, 
                    8, 
                    0  
                    );
                line( 
                    regions, 
                    Point(xC-b, yC),
                    Point(xC+b, yC),
                    Scalar(0,0,0),
                    1, 
                    8, 
                    0  
                    );
            }
            else{
                line( 
                    regions, 
                    Point(xC+a, yC+a*tan(theta)),
                    Point(xC-a, yC-a*tan(theta)),
                    Scalar(0,0,0),
                    1, 
                    8, 
                    0  
                    );
                line( 
                    regions, 
                    Point(xC-a*tan(theta), yC+a),
                    Point(xC+a*tan(theta), yC-a),
                    Scalar(0,0,0),
                    1, 
                    8, 
                    0  
                    );

            }
        }
    }
    cout<<endl;  
}


// Convert CRawImage to Mat
void rawToMat( Mat &destImage, CRawImage* sourceImage){	
	uchar *pointerImage = destImage.ptr(0);
	
	for (int i = 0; i < 240*320; i++)
	{
		pointerImage[3*i] = sourceImage->data[3*i+2];
		pointerImage[3*i+1] = sourceImage->data[3*i+1];
		pointerImage[3*i+2] = sourceImage->data[3*i];
	}
}


void funcion_prueba() {

    cout<<"funcion prueba"<<endl;
    //Despegue
    heli->takeoff();
    usleep(3500000);
    cout<<"takeoff"<<endl;

    //hover
    //heli->setAngles(pitch, roll, yaw, height, hover);
    heli->setAngles(0.0, 0.0, 0.0, 0.0, 1);
    usleep(2000000);
    cout<<"hover"<<endl;

    //Yaw
    //heli->setAngles(pitch, roll, yaw, height, hover);
    heli->setAngles(0.0, 0.0, 20000.0, 0.0, 0.0);
    usleep(1000000);
    cout<<"yaw"<<endl;

    //hover
    //heli->setAngles(pitch, roll, yaw, height, hover);
    heli->setAngles(0.0, 0.0, 0.0, 0.0, 1);
    usleep(2000000);
    cout<<"hover2"<<endl;

    //Pitch
    //heli->setAngles(pitch, roll, yaw, height, hover);
    heli->setAngles(-10000, 0.0, 0.0, 0.0, 0.0);
    usleep(500000);
    cout<<"pitch"<<endl;

}

int main(int argc,char* argv[])
{
	//establishing connection with the quadcopter
	heli = new CHeli();
	
	//this class holds the image from the drone	
	image = new CRawImage(320,240);
	
	// Initial values for control	
    pitch = roll = yaw = height = 0.0;
    joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

	// Destination OpenCV Mat	
	Mat currentImage = Mat(240, 320, CV_8UC3);
    //Filters and Segmentation
    Mat filteredHsvImage,blobDetection,colouredRegions, dilateF;  

    while (stop == false){
        //image is captured
        heli->renewImage(image);
        // Copy to OpenCV Mat
        rawToMat(currentImage, image);

        imageHSVFunction(currentImage, filteredHsvImage);  
        dilateFilter(filteredHsvImage, dilateF);             
        blobColouring(dilateF,blobDetection,colouredRegions);
        imshow("Original"         , currentImage);
        imshow("HSV Filter"       , filteredHsvImage);
        imshow("Dilate Filter"    , dilateF);
        imshow("Blob Colouring"   , blobDetection);
        imshow("Coloured Regions ", colouredRegions);

        char key = waitKey(5);
        switch (key) {
            case 27: stop = true; break;
            default: pitch = roll = yaw = height = 0.0;
        }
        usleep(15000);
    }

    // Initialize joystick
    /*
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
    useJoystick = SDL_NumJoysticks() > 0;
    if (useJoystick)
    {
        SDL_JoystickClose(m_joystick);
        m_joystick = SDL_JoystickOpen(0);
    }

    while (stop == false)
    {

        // Clear the console
        printf("\033[2J\033[1;1H");

        if (useJoystick)
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            joypadRoll = SDL_JoystickGetAxis(m_joystick, 2);
            joypadPitch = SDL_JoystickGetAxis(m_joystick, 3);
            joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
            joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
            joypadTakeOff = SDL_JoystickGetButton(m_joystick, 1);
            joypadLand = SDL_JoystickGetButton(m_joystick, 2);
            joypadHover = SDL_JoystickGetButton(m_joystick, 0);
        }

        // prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
        printf("===================== Parrot Basic Example =====================\n\n");
        fprintf(stdout, "Angles  : %.2lf %.2lf %.2lf \n", helidata.phi, helidata.psi, helidata.theta);
        fprintf(stdout, "Speeds  : %.2lf %.2lf %.2lf \n", helidata.vx, helidata.vy, helidata.vz);
        fprintf(stdout, "Battery : %.0lf \n", helidata.battery);
        fprintf(stdout, "Hover   : %d \n", hover);
        fprintf(stdout, "Joypad  : %d \n", useJoystick ? 1 : 0);
        fprintf(stdout, "  Roll    : %d \n", joypadRoll);
        fprintf(stdout, "  Pitch   : %d \n", joypadPitch);
        fprintf(stdout, "  Yaw     : %d \n", joypadYaw);
        fprintf(stdout, "  V.S.    : %d \n", joypadVerticalSpeed);
        fprintf(stdout, "  TakeOff : %d \n", joypadTakeOff);
        fprintf(stdout, "  Land    : %d \n", joypadLand);
        fprintf(stdout, "Navigating with Joystick: %d \n", navigatedWithJoystick ? 1 : 0);

	cout<<"Prueba Luis"<<endl;
	
		//image is captured
		heli->renewImage(image);

		// Copy to OpenCV Mat
		rawToMat(currentImage, image);
		imshow("ParrotCam", currentImage);

        char key = waitKey(5);
		switch (key) {
			case 'a': yaw = -20000.0; break;
			case 'd': yaw = 20000.0; break;
			case 'w': height = -20000.0; break;
			case 's': height = 20000.0; break;
			case 'q': heli->takeoff(); break;
			case 'e': heli->land(); break;
			case 'z': heli->switchCamera(0); break;
			case 'x': heli->switchCamera(1); break;
			case 'c': heli->switchCamera(2); break;
			case 'v': heli->switchCamera(3); break;
			case 'j': roll = -20000.0; break;
			case 'l': roll = 20000.0; break;
			case 'i': pitch = -20000.0; break;
			case 'k': pitch = 20000.0; break;
            case 'h': hover = (hover + 1) % 2; break;
            case 27: stop = true; break;
			case 'm': funcion_prueba(); break;
            default: pitch = roll = yaw = height = 0.0;
		}

        if (joypadTakeOff) {
            heli->takeoff();
        }
        if (joypadLand) {
            heli->land();
        }
        hover = joypadHover ? 1 : 0;

        //setting the drone angles
        if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
        {
            heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
            navigatedWithJoystick = true;
        }
        else
        {
            heli->setAngles(pitch, roll, yaw, height, hover);
            navigatedWithJoystick = false;
        }

        usleep(15000);
	}
	
	heli->land();
    SDL_JoystickClose(m_joystick);
    */
    waitKey(0);
    delete heli;
	delete image;
	return 0;
}
