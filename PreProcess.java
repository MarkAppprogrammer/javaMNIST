/*
 * Mark Agib
 * 4/28/24
 * Final
 */

package Final;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class PreProcess {
    public static double[][] processImages(int numImagesToRead) throws FileNotFoundException, IOException {
        String filePath = "C:\\Users\\Mark\\APCSA\\Final\\samples\\train-images.idx3-ubyte";
    
        try (DataInputStream inputStream = new DataInputStream(new FileInputStream(filePath))) {
            int magicNumber = inputStream.readInt();
            if (magicNumber != 0x00000803) {
                System.err.println("Invalid magic number. This may not be a valid image file.");
                return null;
            }
    
            inputStream.readInt();
            int numRows = inputStream.readInt();
            int numColumns = inputStream.readInt();
            System.out.println("Processing " + numImagesToRead + " " + numRows + "x" + numColumns + " images");
    
            int[][] images = new int[numImagesToRead][numRows * numColumns];
            for (int i = 0; i < numImagesToRead; i++) {
                for (int j = 0; j < images[i].length; j++) {
                    images[i][j] = inputStream.readUnsignedByte();
                }
                if (i % 10 == 0) {
                    updateProgress(i, numImagesToRead);
                }
            }
    
            double[][] orderedImages = new double[numImagesToRead][numRows * numColumns];
            for (int i = 0; i < numImagesToRead; i++) {
                orderedImages[i] = minMaxNormalization(images[i]);
            }
            System.out.println("");
            System.out.println("Finished processing images!");
            return transpose(orderedImages);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    

    public static double[][] processLabels(int numLabelsToRead) {
        String labelFilePath = "C:\\Users\\Mark\\APCSA\\Final\\samples\\train-labels.idx1-ubyte";
        try (DataInputStream inputStream = new DataInputStream(new FileInputStream(labelFilePath))) {
            int magicNumber = inputStream.readInt();
            if (magicNumber != 0x00000801) {
                System.err.println("Invalid magic number. This may not be a valid labels file.");
                return null;
            }

            int numLabels = inputStream.readInt(); // Read the number of labels
            if (numLabels < numLabelsToRead) {
                System.err.println("Requested more labels than available.");
                return null;
            }
            
            byte[] labels = new byte[numLabelsToRead];
            inputStream.readFully(labels);

            System.out.println("Processing " + numLabelsToRead + " labels");

            double[] orderedLabels = new double[numLabelsToRead];
            for (int i = 0; i < numLabelsToRead; i++) {
                orderedLabels[i] = (double) labels[i];
            }
            double[] categories = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

            System.out.println("Finished processing labels!");
            return oneHotEncode(orderedLabels, categories);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static byte[] flatten(byte[][] image, int numRows, int numColumns) {
        byte[] temp = new byte[numColumns * numRows];

        int i = 0;
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numColumns; col++) {
                temp[i] = image[row][col];
                i++;
            }
        }    
        return temp;
    }

    public static double[] minMaxNormalization(byte[] image) {
        byte minPixelValue = Byte.MAX_VALUE;
        byte maxPixelValue = Byte.MIN_VALUE;
        for (byte pixelValue : image) {
            if (pixelValue < minPixelValue) {
                minPixelValue = pixelValue;
            }
            if (pixelValue > maxPixelValue) {
                maxPixelValue = pixelValue;
            }
        }

        double[] newImages = new double[image.length];

        for (int i = 0; i < image.length; i++) {
            newImages[i] = ((image[i] & 0xFF - minPixelValue) / (double) (maxPixelValue & 0xFF - minPixelValue));
        }

        return newImages;
    }

    public static double[] minMaxNormalization(int[] image) {
        int minPixelValue = 0;
        int maxPixelValue = 255;
        for (int pixelValue : image) {
            if (pixelValue < minPixelValue) {
                minPixelValue = pixelValue;
            }
            if (pixelValue > maxPixelValue) {
                maxPixelValue = pixelValue;
            }
        }

        if (minPixelValue == maxPixelValue) {
            double[] newImages = new double[image.length];
            for (int i = 0; i < image.length; i++) {
                newImages[i] = 0.5; 
            }
            return newImages;
        }

        double[] newImages = new double[image.length];

        for (int i = 0; i < image.length; i++) {
            newImages[i] = ((image[i] - minPixelValue) / (double) (maxPixelValue - minPixelValue));
        }

        return newImages; 
    }

    public static double[][] oneHotEncode(double[] labels, double[] categories) {
        double[][] result = new double[categories.length][labels.length];

        for (int i = 0; i < categories.length; i++) {
            for (int j = 0; j < labels.length; j++) {
                if (labels[j] == categories[i]) {
                    result[i][j] = 1.0;
                }
                else {
                    result[i][j] = 0.0;
                }
            }
        }

        return result;
    }

    public static void updateProgress(int currentStep, int totalSteps) {
        double progress = (double) currentStep / totalSteps;
        int barLength = 100;

        System.out.print("\r[");
        int progressChars = (int) (progress * barLength);
        for (int i = 0; i < barLength; i++) {
            if (i < progressChars) {
                System.out.print("=");
            } else {
                System.out.print(" ");
            }
        }
        System.out.printf("] %.2f%%", progress * 100);
    }

    // public static void accuracy() {
    // }

    public static double[][] dot(double[][] matrixA, double[][] matrixB) {
        double[][] newMatrix = new double[matrixA.length][matrixB[0].length];

        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                for (int k = 0; k < matrixB.length; k++) {
                    newMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        return newMatrix;
    }

    public static double[][] matrixCoefficientMultiplication(double[][] matrix, double coefficent) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                newMatrix[i][j] = matrix[i][j] * coefficent;
            }
        }

        return newMatrix;
    }

    public static double[][] matrixOperations(double[][] matrixA, double[][] matrixB, boolean subtraction) {
        double[][] newMatrix = new double[matrixA.length][matrixA[0].length];

        double[][] newMatrixB;
        if (subtraction) {
            newMatrixB = matrixCoefficientMultiplication(matrixB, -1.0);
        }
        else {
            newMatrixB = matrixB;
        }
        
        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                newMatrix[i][j] = matrixA[i][j] + newMatrixB[i][j];
            }
        }

        return newMatrix;
    }

    public static double[][] matrixExp(double[][] matrix) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < newMatrix.length; i++) {
            for (int j = 0; j < newMatrix[i].length; j++) {
                newMatrix[i][j] = Math.exp(matrix[i][j]);
            }
        }

        return newMatrix;
    }

    public static double sum(double[][] matrix) {
        double sum = 0;
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j];
            }
        }

        return sum;
    }

    public static double[][] sumSecondAxis(double[][] array) {
        double[][] sums = new double[array.length][1];

        for (int i = 0; i < array.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < array[i].length; j++) {
                sum += array[i][j];
            }
            sums[i][0] = sum;
        }

        return sums;
    }

    public static double[][] reshape(double[][] matrix) {
        double[][] newMatrix = new double[matrix.length * matrix[0].length][1];

        int k = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[k][0] = matrix[i][j];
                k++;
            }
        }

        return newMatrix;
    }

    public static double[] reshape(double[][] matrix, int numColumns) {
        double[] newMatrix = new double[matrix.length * matrix[0].length];

        int k = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newMatrix[k] = matrix[i][j];
                k++;
            }
        }

        return newMatrix;
    }

    public static double[][] transpose(double[][] array) {
        int length = array.length;
        int imageSize = array[0].length;
        double[][] transposedMatrix = new double[imageSize][length];

        for (int i = 0; i < imageSize; i++) {
            for (int j = 0; j < length; j++) {
                transposedMatrix[i][j] = array[j][i];
            }
        }

        return transposedMatrix;
    }

    public static double[][] copyAcross(double[][] matrix, int numColumns) {
        double[][] result = new double[matrix.length][numColumns];

        for (int i = 0; i < matrix.length; i++) {
            double value = matrix[i][0]; 
            for (int j = 0; j < numColumns; j++) {
                result[i][j] = value;
            }
        }

        return result;
    }  
    
    public static double[][] elementWiseMult(double[][] matrixA, double[][] matrixB) {
        double[][] newMatrix = new double[matrixA.length][matrixA[0].length];

        for (int i = 0; i < matrixA.length; i++) {
            for (int j = 0; j < matrixA[i].length; j++) {
                newMatrix[i][j] = matrixA[i][j] * matrixB[i][j];
            }
        }

        return newMatrix;
    }
}

