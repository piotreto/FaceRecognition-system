#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <filesystem>


namespace fs = std::filesystem;
using namespace cv;
using namespace cv::face;
using namespace std;

const int SAMPLE_WIDTH = 128;
const int SAMPLE_HEIGHT = 128;



void getNewPerson()
{
    CascadeClassifier faceCascade;
    faceCascade.load("resources/haarcascade_frontalface_alt.xml");

    if (faceCascade.empty())
    {
        cout << "I couldnt open xml file" << endl;
        return;
    }
    
    int ID;
    cout << "Enter your ID: ";
    cin >> ID;

    stringstream ssfn;
    ssfn << "faces/" << ID;
    string dir_path = ssfn.str();

    ssfn.str(string());
    
    if (mkdir(dir_path.c_str(), 0777) < 0)
    {
        cout << "Error while creating folder..." << endl;
        return;
    };

    int idx = 0;

    VideoCapture cap(0);
    Mat frame;
    Mat crop, result;

    namedWindow("Image", 1);

    while (true)
    {
        while(true){
            cap.read(frame);
            vector<Rect> faces;
            faceCascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));

            for (int i = 0; i < faces.size(); i++)
            {
                crop = frame(faces[i]);
                cvtColor(crop, crop, COLOR_BGR2GRAY);
                resize(crop, result, Size(SAMPLE_HEIGHT, SAMPLE_WIDTH),  1, 1, INTER_CUBIC);
                rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
            }
            imshow("Image", frame);
            char c = waitKey(10);
            if(c == 'c'){
                break;
            }
        }

	    ssfn << dir_path << "/" << idx << ".bmp";
        string path = ssfn.str();
        cout << path << endl;
	    imwrite(path, result);
        ssfn.str(""); // clearing stringstream
        idx += 1;
        if (idx == 10)
        {
            break;
        }
    }
    destroyWindow("Image");
}

void readData(vector<Mat>& images, vector<int>& labels)
{
    string path = "faces";
    for(const auto & dir : fs::directory_iterator(path)) {
        int label = stoi(dir.path().filename());
        for(const auto & file: fs::directory_iterator(dir))
        {
            labels.push_back(label);
            Mat image = imread(file.path(), 0);
            images.push_back(imread(file.path(), 0));
        }
    }
}

void trainModel()
{   
    cout << "Training model...." << endl;

    vector<Mat> images;
    vector<int> labels;

    readData(images, labels);

    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    model->save("resources/trainedModel.yml");

    cout << "Training finished...." << endl;
}

void detectPeople(){
    
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->read("resources/trainedModel.yml");

    CascadeClassifier faceCascade;
    faceCascade.load("resources/haarcascade_frontalface_alt.xml");

    if (faceCascade.empty())
    {
        cout << "I couldnt open xml file" << endl;
        return;
    }

    namedWindow("Image", 1);
    Mat frame, crop, cropResized;
    VideoCapture cap(0);
    while(true)
    {
        cap.read(frame);
        vector<Rect> faces;
        faceCascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
        for (int i = 0; i < faces.size(); i++)
        {
            crop = frame(faces[i]);
            cvtColor(crop, crop, COLOR_BGR2GRAY);
            resize(crop, cropResized, Size(SAMPLE_HEIGHT, SAMPLE_WIDTH), 1, 1, INTER_CUBIC);
            rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
            int label = -1;
            double confidence = 0;
            model->predict(cropResized, label, confidence);
            cout << confidence << endl;
            int pos_x = std::max(faces[i].tl().x - 10, 0);
		    int pos_y = std::max(faces[i].tl().y - 10, 0);
            
            putText(frame, to_string(label), Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
        }
        imshow("Image", frame);
        char c = waitKey(1);
        if(c > 0){
            break;
        }
    }
    destroyWindow("Image");

}


int main()
{
    // getNewPerson();
    // trainModel();
    detectPeople();
}