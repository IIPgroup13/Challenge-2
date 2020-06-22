//*********************************************
// Example Code for Interactive Intelligent Products
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
import ddf.minim.analysis.*;
import ddf.minim.*;

Minim minim;
AudioInput in;
FFT fft;

int streamSize = 500;
float sampleRate = 44100/5;
int numBins = 1025;
int bufferSize = (numBins-1)*2;
//FFT parameters
float[][] FFTHist;
final int LOW_THLD = 1; //low threshold of band-pass frequencies
final int HIGH_THLD = 200; //high threshold of band-pass frequencies 
int numBands = HIGH_THLD-LOW_THLD+1; //number of feature
float[] modeArray = new float[streamSize]; //classification to show
float[] thldArray = new float[streamSize]; //diff calculation: substract

//segmentation parameters
float energyMax = 0;
float energyThld = 5;
float[] energyHist = new float[streamSize]; //history data to show//segmentation parameters

//window
int windowSize = 15; //The size of data window
float[][] windowArray = new float[numBands][windowSize]; //data window collection
boolean b_sampling = false; //flag to keep data collection non-preemptive
int sampleCnt = 0; //counter of samples

//Statistical Features
float[] windowM = new float[numBands]; //mean
float[] windowSD = new float[numBands]; //standard deviation

//Save
Table csvData;
boolean b_saveCSV = false;
String dataSetName = "SentimentTrain"; 
String[] attrNames;
boolean[] attrIsNominal;
int labelIndex = 0;

String lastPredY = "";

//multi model arrays
ArrayList<Attribute>[] attributes = new ArrayList[1];
Instances[] instances = new Instances[1];
Classifier[] classifiers = new Classifier[2];

void setDataType() {
  attrNames =  new String[numBands+1];
  attrIsNominal = new boolean[numBands+1];
  for (int j = 0; j < numBands; j++) {
    attrNames[j] = "f_"+j;
    attrIsNominal[j] = false;
  }
  attrNames[numBands] = "label";
  attrIsNominal[numBands] = true;
}


void setup()
{
  size(1200, 800);
  // setup audio input
  minim = new Minim(this);
  in = minim.getLineIn(Minim.MONO, bufferSize, sampleRate);
  fft = new FFT(in.bufferSize(), in.sampleRate());
  fft.window(FFT.NONE);
  FFTHist = new float[numBands][streamSize]; //history data to show
  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }
  /*
  loadTrainARFF(dataset="testData.arff"); //load a ARFF dataset
  loadModel(model="LinearSVCSentiment.model"); //load a pretrained model.
  */
  instances[0] = loadTrainARFFToInstances(dataset="testSentiment.arff");
  attributes[0] = loadAttributesFromInstances(instances[0]);
  classifiers[0] = loadModelToClassifier(model="PolySVCSentiment_best.model"); //load a pretrained model.
  classifiers[1] = loadModelToClassifier(model="LinearSVCSentiment_best.model"); //load a pretrained model.
  //classifiers[2] = loadModelToClassifier(model="RBFSVCSentiment_best.model"); //load a pretrained model.
  loadTestARFF(dataset="testSentiment.arff");//load a ARFF dataset
  evaluateTestSet(classifiers[0],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
  evaluateTestSet(classifiers[1],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
  //evaluateTestSet(classifiers[2],test,isRegression = false, showEvalDetails=true);  //5-fold cross validation
}

void draw()
{
  background(255);
  fft.forward(in.mix.toArray());

  float[] X = new float[numBands]; //Form a feature vector X;

  energyMax = 0; //reset the measurement of energySum
  for (int i = 0; i < HIGH_THLD-LOW_THLD; i++) {
    float x = fft.getBand(i+LOW_THLD);
    if (x>energyMax) energyMax = x;
    if (b_sampling == true) {
      if (x>X[i]) X[i] = x; //simple windowed max
      windowArray[i][sampleCnt-1] = x; //windowed statistics
    }
  }

  if (energyMax>energyThld) {
    if (b_sampling == false) { //if not sampling
      b_sampling = true; //do sampling
      sampleCnt = 0; //reset the counter
      for (int j = 0; j < numBands; j++) {
        X[j] = 0; //reset the feature vector
        for (int k = 0; k < windowSize; k++) {
          (windowArray[j])[k] = 0; //reset the window
        }
      }
    }
  } 

  if (b_sampling == true) {
    ++sampleCnt;
    if (sampleCnt == windowSize) {
      for (int j = 0; j < numBands; j++) {
        windowM[j] = Descriptive.mean(windowArray[j]); //mean
        windowSD[j] = Descriptive.std(windowArray[j], true); //standard deviation
        X[j] = max(windowArray[j]);
      }
      b_sampling = false;
      String[] Y = new String[classifiers.length];
      double[] yID = new double[classifiers.length];
      for(int i = 0 ; i < classifiers.length ; i++){
        Y[i] = getPrediction(X, classifiers[i], attributes[0],instances[0]);
        yID[i] = getPredictionIndex(X, classifiers[i], attributes[0]);
        lastPredY = Y[i];
          for (int n = 0; n < windowSize; n++) {
          appendArrayTail(modeArray, (float)yID[i]);
        }
      }
      /*
      lastPredY = getPrediction(X)
      double yID = getPredictionIndex(X);
      for (int n = 0; n < windowSize; n++) {
        appendArrayTail(modeArray, (float)yID);
      }
      */
    }
  } else {
    appendArrayTail(modeArray, -1); //the class is null without mouse pressed.
  }

  String Z = lastPredY;
  String B = "nothing";
  int c = 0;

  if (Z.equals("A")) {
    B = "yes confident";
    c = 0;
    background(0, 250, 0);
  }
  if (Z.equals("B")) {
    B = "yes doubtful";
    c = 50;
    background(0, 100, 0);
  }
  if (Z.equals("C")) {
    B = "no confident";
    c = 200;
    background(250, 0, 0);
  }
  if (Z.equals("D")) {
    B = "no doubtful";
    c = 255;
    background(100, 0, 0);
  }

  showInfo("I think you are saying "+B, width/2, height/2, c, 50);

  if (b_saveCSV) {
    saveCSV(dataSetName, csvData);
    saveARFF(dataSetName, csvData);
    b_saveCSV = false;
  }
}

void stop()
{
  // always close Minim audio classes when you finish with them
  in.close();
  minim.stop();
  super.stop();
}
