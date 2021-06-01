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

// size of images in a base
const int SAMPLE_WIDTH = 128;
const int SAMPLE_HEIGHT = 128;



void getNewPerson()
{
    // firstly we try to load model for detecting faces
    CascadeClassifier faceCascade;
    faceCascade.load("resources/haarcascade_frontalface_alt.xml");

    if (faceCascade.empty())
    {
        cout << "I couldnt open xml file" << endl;
        return;
    }
    
    // asking person for ID
    int ID;
    cout << "Enter your ID: ";
    cin >> ID;

    // creating folder for a new person in faces directory
    stringstream ssfn;
    ssfn << "faces/" << ID;
    string dir_path = ssfn.str();

    ssfn.str(string());
    
    if (mkdir(dir_path.c_str(), 0777) < 0)
    {
        cout << "Error while creating folder..." << endl;
        return;
    };

    cout << "Okay, now you have to focus. Your camera is open, make sure that it detect your face. When purple rectangle "
    << "occurs on your face and you are ready, click \"c\" on your keyboard in order to take a photo." << endl
    << "For better performance change posittions, try to take photo close to camera and a liitle futher" << endl;
    cout << "You have to take 10photos." << endl;
    int idx = 1;

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

            for (auto & face : faces)
            {
                // cropping and resizing face
                crop = frame(face);
                cvtColor(crop, crop, COLOR_BGR2GRAY);
                resize(crop, result, Size(SAMPLE_HEIGHT, SAMPLE_WIDTH),  1, 1, INTER_CUBIC);
                rectangle(frame, face.tl(), face.br(), Scalar(255, 0, 255), 3);
            }
            imshow("Image", frame);

            // when user click 'c' we break this loop and save processed face to the base
            char c = waitKey(10);
            if(c == 'c'){
                break;
            }
        }

        // saving picture
	    ssfn << dir_path << "/" << idx << ".bmp";
        string path = ssfn.str();
        cout << "Nice one! " <<  "You have " << 10 - idx << " photos to take left."<< endl;
	    imwrite(path, result);
        ssfn.str(""); // clearing stringstream
        idx += 1;
        if (idx == 11)
        {
            break;
        }
    }
    destroyWindow("Image");
}

// function for reading and storing data 
void readData(vector<Mat>& images, vector<int>& labels)
{
    // its very simple, we just go into "faces" directory
    // open every single directory which belongs to a single person
    // then we save all images and appropriate label which is name of a folder
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

// function for training model
void trainModel()
{   

    cout << "Training model...." << endl;

    vector<Mat> images;
    vector<int> labels;

    // firtsly we have to read our data and store it in vectors
    readData(images, labels);

    // for training and predcition e use a eigenfaces method
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    // we save our trained model in resources directory
    model->save("resources/trainedModel.yml");

    cout << "Training finished...." << endl;
}

// function for detecting and recognising people
void recognisePeople(){
    
    // creating and reading our prepared model
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->read("resources/trainedModel.yml");

    // model for detecting faces
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

        // for every detected face by our detector
        // we process detected area and using our trained model we can predicte a label for this area(face)
        for (auto & face : faces)
        {
            crop = frame(face);
            cvtColor(crop, crop, COLOR_BGR2GRAY);
            resize(crop, cropResized, Size(SAMPLE_HEIGHT, SAMPLE_WIDTH), 1, 1, INTER_CUBIC);
            rectangle(frame, face.tl(), face.br(), Scalar(255, 0, 255), 3);
            int label = -1;
            double confidence = 0;
            model->predict(cropResized, label, confidence);
            int pos_x = std::max(face.tl().x - 10, 0);
		    int pos_y = std::max(face.tl().y - 10, 0);
            
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
    cout << "Hello in FaceRecognition system" << endl;
    cout << "What would you like to do?" << endl;
    
    cout << "1. Add new face to the base." << endl;
    cout << "2. Regonise faces using your camera." << endl;
    int choice;
    cout << "What is your choice?" << endl;
    cin >> choice;
    switch (choice)
    {
    case 1:
        getNewPerson();
        trainModel();
        break;
    case 2:
        recognisePeople();
        break;
    default:
        cout << "I dont know whats on your mind, see you later.." << endl;
    }

    return 0;
    
}