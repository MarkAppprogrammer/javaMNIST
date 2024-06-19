package Final;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

class ActivationFunction {
    // possibly swith to rmsprop for better op
    // switch to leaky or parametric relu
    public static double[][] sigmoid(double[][] matrix) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];
    
        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                newMatrix[i][j] = 1 / (1 + Math.exp(-matrix[i][j]));
            }
        }
    
        return newMatrix;
    }
    
    public static double[][] sigmoidDerivative(double[][] matrix) {
        double[][] sigmoidMatrix = sigmoid(matrix);  
        double[][] derivative = new double[matrix.length][matrix[0].length];
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                double sigmoidValue = sigmoidMatrix[i][j];
                derivative[i][j] = sigmoidValue * (1 - sigmoidValue);
            }
        }
        
        return derivative;
    }
    

    // public static double sigmoidDerivative(double x) {
    //     return (Math.exp(-x) / Math.pow(1 + Math.exp(-x)))
    // }

    public static double[][] rectifiedLinearUnit(double[][] matrix) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                if (matrix[i][j] >= 0.0) {
                    newMatrix[i][j] = matrix[i][j];
                }
                else {
                    newMatrix[i][j] = 0.01 * matrix[i][j];
                }
            }
        }

        return newMatrix;
    }

    public static double[][] deravtiveRectifiedLinearUnit(double[][] matrix) {
        double[][] derivative = new double[matrix.length][matrix[0].length];
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > 0) {
                    derivative[i][j] = 1;
                } else {
                    derivative[i][j] = 0.01;
                }
            }
        }
        
        return derivative;
    }
    
    
    public static double[][] softMax(double[][] matrix) {
        double[][] softmax = new double[matrix.length][matrix[0].length];
        
        for (int i = 0; i < matrix.length; i++) {
            double sum = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                sum += Math.exp(matrix[i][j]);
            }
            for (int j = 0; j < matrix[i].length; j++) {
                softmax[i][j] = Math.exp(matrix[i][j]) / sum;
            }
        }
        
        return softmax;
    }    
}

class LossFunction {
    public static double meanSquaredError(double[][] actual, double[][] predictions) {
        int numClasses = actual.length;
        int numExamples = actual[0].length;
        double sumSquaredError = 0;
    
        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < numClasses; j++) {
                double error = actual[j][i] - predictions[j][i]; 
                sumSquaredError += Math.pow(error, 2); 
            }
        }
    
        double mse = sumSquaredError / (numClasses * numExamples);
        return mse;
    }
    
    public static double crossEntropy(double[][] probabilities, double[][] oneHotEncoded) {
        int numExamples = oneHotEncoded.length; // Number of examples
        int numClasses = oneHotEncoded[0].length; // Number of classes
        double loss = 0;
        double epsilon = 1e-15; // Small value to avoid log(0)
        double tolerance = 1e-9; // Tolerance for floating-point precision
    
        // Check probabilities are valid
        for (int i = 0; i < numClasses; i++) {
            double sum = 0;
            for (int j = 0; j < numExamples; j++) {
                if (probabilities[i][j] < 0 || probabilities[i][j] > 1) {
                    throw new IllegalArgumentException("Probabilities must be between 0 and 1.");
                }
                sum += probabilities[i][j];
            }
            if (Math.abs(sum - 1) > tolerance) {
                System.out.println(sum);
                throw new IllegalArgumentException("Probabilities for each example must sum to 1.");
            }
        }
    
        // Check one-hot encoding is valid
        for (int i = 0; i < numClasses; i++) {
            int count = 0;
            for (int j = 0; j < numExamples; j++) {
                if (oneHotEncoded[i][j] == 1) {
                    count++;
                } else if (oneHotEncoded[i][j] != 0) {
                    throw new IllegalArgumentException("One-hot encoding must be 0 or 1.");
                }
            }
            if (count != 1) {
                //System.out.println(i);
                throw new IllegalArgumentException("Each example in one-hot encoding must have exactly one class labeled as 1.");
            }
        }
    
        // Calculate the loss
        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (oneHotEncoded[i][j] == 1) {
                    loss += Math.log(probabilities[i][j] + epsilon);
                }
            }
        }
    
        return -loss / numExamples; // Return the average loss
    } 
}

class Neuron {
    private double[] weights;
    private double bias;

    public Neuron(int numInputs, int numOutputs) {
        Random rand = new Random();
        double initWeightRange = Math.sqrt(6.0 / (numInputs + numOutputs)); // Xavier initialization
        weights = new double[numInputs];
        for (int i = 0; i < numInputs; i++) {
            weights[i] = rand.nextDouble() * 2 * initWeightRange - initWeightRange;
        }
        bias = 0.0; // Initialize bias to zero
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public void updateWeights(double[] weights) {
        this.weights = weights;
    }

    public void updateBias(double bias) {
        this.bias = bias;
    }
}

class Layer {
    private Neuron[] neurons;

    public Layer(int numNeurons, int numInputsPerNeuron, int numOutputNeurons) {
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(numInputsPerNeuron, numOutputNeurons);
        }
    }

    // public double[][] calculateOutputs(double[] inputs, int numberOfImages) {
    //     double[][] outputs = new double[numberOfImages][neurons.length];

    //     for (int r = 0; r < neurons.length; r++) {
    //         for (int c = 0; c < neurons.length; c++) {
    //             outputs[r][c] = neurons[c].calculateOutput(inputs);
    //         }
    //     }
        
    //     return outputs;
    // }

    // public double[] calculateOutputs(double[] inputs, double[] target) {
    //     double[] classProb = new double[inputs.length];
    //     for (int i = 0; i < inputs.length; i++) {
    //         classProb[i] = ActivationFunction.softMax(inputs[i], inputs);
    //     }
    //     return classProb;
    // }

    public static int classify(double[] classProb) {
        double highest = classProb[0];
        int highestIndex = 0;

        for (int j = 0; j < classProb.length; j++) {
            if (highest < classProb[j]) {
                highest = classProb[j];
                highestIndex = j;
            }
        }
        
        return highestIndex;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public void updateNeurons(double[][] weights, double[][] bias) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].updateWeights(weights[i]);
            neurons[i].updateBias(bias[i][0]);
        }
    }
}

class NeuralNetwork {
    private Layer hiddenLayer;
    private Layer outputLayer;

    public NeuralNetwork(int numInputNeurons, int numHiddenNeurons, int numOutputNeurons) {
        hiddenLayer = new Layer(numHiddenNeurons, numInputNeurons, numOutputNeurons);
        outputLayer = new Layer(numOutputNeurons, numHiddenNeurons, numOutputNeurons);
    }

    // public double[] forwardPass(double[] inputs) {
    //     double[] hiddenOutputs = hiddenLayer.calculateOutputs(inputs);
    //     return outputLayer.calculateOutputs(hiddenOutputs); //fix
    // }

    public int[] getPredctions(double[][] percentageMatrix) {
        int[] predictions = new int[percentageMatrix[0].length];
        double[][] tempMatrix = PreProcess.transpose(percentageMatrix);

        for (int i = 0; i < tempMatrix.length; i++) {
            double highest = tempMatrix[i][0];
            int highestIndex = 0;

            for (int j = 0; j < tempMatrix[i].length; j++) {
                if (tempMatrix[i][j] > highest) {
                    highest = tempMatrix[i][j];
                    highestIndex = j;
                }
            }

            predictions[i] = highestIndex;
        }
        
        return predictions;
    }

    public double train(double[][] inputs, double[][] targets, double learningRate, int numberOfImages, int epoch) {
        double[][] hiddenWeights = new double[targets.length][inputs.length];
        double[][] hiddenBiases = new double[targets.length][1];
        Neuron[] hiddenNeurons = hiddenLayer.getNeurons();
        for (int i = 0; i < hiddenNeurons.length; i++) {
            hiddenWeights[i] = hiddenNeurons[i].getWeights();
            hiddenBiases[i][0] = hiddenNeurons[i].getBias();
        }
        
        double[][] hiddenInputs = PreProcess.matrixOperations(PreProcess.dot(hiddenWeights, inputs), PreProcess.copyAcross(hiddenBiases, numberOfImages), false);
        double[][] hiddenOutputs = ActivationFunction.sigmoid(hiddenInputs);

        double[][] outputWeights = new double[targets.length][inputs.length];
        double[][] outputBiases = new double[targets.length][1];
        Neuron[] outputNeurons = outputLayer.getNeurons();
        for (int i = 0; i < outputNeurons.length; i++) {
            outputWeights[i] = outputNeurons[i].getWeights();
            outputBiases[i][0] = outputNeurons[i].getBias();
        }

        double[][] outputInputs = PreProcess.matrixOperations(PreProcess.dot(outputWeights, hiddenOutputs), PreProcess.copyAcross(outputBiases, numberOfImages), false);
        double[][] actualOutputs = ActivationFunction.softMax(outputInputs);

        // back prob down here
        // implemnt MSE

        // double[][] outputErrors = PreProcess.matrixCoefficientMultiplication(PreProcess.matrixOperations(actualOutputs, targets, true), (2.0 / numberOfImages));
        // double[][] outputErrorWeights = PreProcess.dot(PreProcess.transpose(hiddenOutputs)); (JUST REMOVE COFFIENCT MULT ON WEIGHTS AND CHECK)
        // add elemnt wise mult between actvation function and current.
        double[][] outputErrors = PreProcess.matrixCoefficientMultiplication(PreProcess.matrixOperations(actualOutputs, targets, true), (2.0 / numberOfImages));
        
        double[][] deravtiveOutputWeigths = PreProcess.dot(outputErrors, PreProcess.transpose(hiddenOutputs));
        double[][] deravtiveOutputBiases = PreProcess.sumSecondAxis(outputErrors);

        double[][] hiddenErrors = PreProcess.elementWiseMult(PreProcess.dot(PreProcess.transpose(deravtiveOutputWeigths), outputErrors), ActivationFunction.sigmoidDerivative(hiddenOutputs));

        double[][] deravtiveHiddenWeigths = PreProcess.dot(hiddenErrors, PreProcess.transpose(inputs));
        double[][] deravtiveHiddenBiases = PreProcess.sumSecondAxis(hiddenErrors);

        // updating time :)
        outputLayer.updateNeurons(PreProcess.matrixOperations(outputWeights, PreProcess.matrixCoefficientMultiplication(deravtiveOutputWeigths, learningRate), true), PreProcess.matrixOperations(outputBiases, PreProcess.matrixCoefficientMultiplication(deravtiveOutputBiases, learningRate), true));
        hiddenLayer.updateNeurons(PreProcess.matrixOperations(hiddenWeights, PreProcess.matrixCoefficientMultiplication(deravtiveHiddenWeigths, learningRate), true), PreProcess.matrixOperations(hiddenBiases, PreProcess.matrixCoefficientMultiplication(deravtiveHiddenBiases, learningRate), true));
        
        double sum = 0;
        if (epoch % 50 == 0) {
            int[] predictions = getPredctions(actualOutputs);
            for (int k = 0; k < predictions.length; k++) {
                if (targets[predictions[k]][k] == 1.0) {
                    sum += 1;
                }
            }

            System.out.println("Loss: " + LossFunction.crossEntropy(actualOutputs, targets));
            
             return (sum / numberOfImages) * 100;
        }

        // double[][] actualOutputs
        // double[][] hiddenOutputs = hiddenLayer.calculateOutputs(inputs, numberOfImages);
        // double[] actualOutputs = outputLayer.calculateOutputs(outputLayer.calculateOutputs(hiddenOutputs), targets);
        // double sum = 0;

        // // all back prob lower
        // double[] outputErrors = new double[actualOutputs.length];
        // for (int i = 0; i < actualOutputs.length; i++) {
        //     outputErrors[i] = actualOutputs[i] - targets[i]; // fix targets use
        // }

        // if (exampleNumber == 59999) {
        //     // targets are one hot encoded and I used them diff
        //     for (int i = 0; i < actualOutputs.length; i++) {
        //         int correctIndex = 0;
        //         for (int j = 0; j < targets.length; j++) {
        //             if (targets[j] == 1.0) {
        //                 correctIndex = j;
        //             }
        //         }
        //         if (Layer.classify(actualOutputs) == correctIndex) {
        //             sum += 1;
        //         }
        //     }

        //     sum = sum / actualOutputs.length;
        // }
        // // above = goog fix below to show changes
        // // deravtive of weights = (1/number of training images) dZ(output errors) (outputs of output layer)
        // Neuron[] outputNeurons = outputLayer.getNeurons();
        // // = new double[outputErrors.length][hiddenOutputs.length]; //(1.0 / outputNeurons.length * hiddenOutputs.length);
        // double[] deravtiveBiases = new double[outputErrors.length];
        // final double averageMaker = (1.0 / numberOfImages);
        //double[][] deravtiveWeigths = PreProcess.matrixCoefficientMultiplication(PreProcess.dot(outputErrors, actualOutputs), averageMaker);
        //for (int i = 0; i < deravtiveWeigths.length; i++) {
        //     deravtiveWeigths[i] = outputErrors[i] * actualOutputs[i] * averageMaker;
        //     double[] outputWeights = outputNeurons[i].getWeights();
        //     double outputBias = outputNeurons[i].getBias();
        //     for (int j = 0; j < outputWeights.length; j++) {
        //         outputWeights[j] += learningRate * outputErrors[i] * ActivationFunction.sigmoidDerivative(outputNeurons[i].getOutput()) * hiddenOutputs[j];
        //     }
        //     outputBias += learningRate * outputErrors[i] * ActivationFunction.sigmoidDerivative(outputNeurons[i].getOutput());
        //}
        

        // // Update weights and biases in hidden layer
        // Neuron[] hiddenNeurons = hiddenLayer.getNeurons();
        // for (int i = 0; i < hiddenNeurons.length; i++) {
        //     double[] hiddenWeights = hiddenNeurons[i].getWeights();
        //     double hiddenBias = hiddenNeurons[i].getBias();
        //     double sum2 = 0;
        //     for (int j = 0; j < outputNeurons.length; j++) {
        //         sum2 += outputErrors[j] * ActivationFunction.sigmoidDerivative(outputNeurons[j].getOutput()) * outputNeurons[j].getWeights()[i];
        //     }
        //     for (int j = 0; j < hiddenWeights.length; j++) {
        //         hiddenWeights[j] += learningRate * sum2 * ActivationFunction.sigmoidDerivative(hiddenNeurons[i].getOutput()) * inputs[j];
        //     }
        //     hiddenBias += learningRate * sum2 * ActivationFunction.sigmoidDerivative(hiddenNeurons[i].getOutput());
        // }

        return sum;
    }
}

public class Main {
    public static void main(String[] args) throws FileNotFoundException, IOException {
        NeuralNetwork neuralNetwork = new NeuralNetwork(784, 10, 10);

        System.out.println("[*] Choose an Option below and enter the number:");
        System.out.println("1. Train");
        System.out.println("2. Test");

        /*
         * TO DO:
         * Parallel Processing
         * using ByteBuffer
         * Adding print statments to show progess 
         * Allow error checking?
         * intliaze arrays outside of loops to save mem
         */

        @SuppressWarnings("resource")
        Scanner input = new Scanner(System.in);
        int option = input.nextInt();

        if (option == 1) {
            double[][] trainingInputs = PreProcess.processImages(4000);
            double[][] trainingOutputs = PreProcess.processLabels(4000);
            int epochs = 401;
            double learningRate = 0.06;
            System.out.println("Starting training");
            for (int i = 0; i < epochs; i++) {
                double accuracy = neuralNetwork.train(trainingInputs, trainingOutputs, learningRate, 4000, i);
                if (i % 50 == 0) {
                    System.out.println("Epoch: " + i + " Accuarcy: " + accuracy);
                }
            }
            // for (int i = 0; i < epochs; i++) {
            //     double accuracy = 0;
            //     for (int l = 0; l < trainingInputs.length; l++) {
            //         currentImage = trainingInputs[l];
            //         currentLabel = trainingOutputs[l];
            //         accuracy = neuralNetwork.train(currentImage, currentLabel, learningRate, l);
            //         if (i % 40 == 0){
            //             PreProcess.updateProgress(l, trainingInputs.length);
            //         }
            //     }
            //     System.out.println("Epoch: " + i + "Accuracy: " + accuracy);
            // }
            // make sure array stuct aligns
        } else if (option == 2) {
            /*
            int correctPredictions = 0;
                for (int i = 0; i < testDataSize; i++) {
                    double[] predictedOutputs = neuralNetwork.forwardPass(testInputs[i]);
                    int predictedLabel = findIndexOfMax(predictedOutputs);
                    if (predictedLabel == trueLabels[i]) { // Assuming trueLabels contain ground truth labels
                        correctPredictions++;
                    }
                }
                double accuracy = (double) correctPredictions / testDataSize;

             */
        } else {
            System.out.println("No second Chances try again next time");
        }

        // for (int i = 0; i < trainingInputs.length; i++) {
        //     double[] inputs = trainingInputs[i];
        //     double[] predictedOutputs = neuralNetwork.forwardPass(inputs);
        //     System.out.println("Input: " + inputs[0] + ", " + inputs[1] + " | Predicted Output: " + predictedOutputs[0]);
        // }
    }
}