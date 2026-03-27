public class Perceptron {
    private double[] weights;
    private double alphaLearningConstant;
    //Przyjmujemy długość wektora wag, próg , próg włączenia perceptronu, oraz stałą uczenia alfa
    Perceptron(int LengthOfTheWeightVector, double alphaLearningConstant){
        weights = new double[LengthOfTheWeightVector + 1]; // +1 bias
        for(int i = 0; i < weights.length; i++){
            weights[i] = (Math.random()*2)-1;
        }
        this.alphaLearningConstant = alphaLearningConstant;
    }

    int classify(double[] data){
        if(data.length != this.weights.length - 1){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        double sum = 0;
        
        // normalne wagi
        for(int i = 0; i < this.weights.length - 1; i++){
            sum += data[i] * weights[i];
        }

        // bias
        sum += weights[weights.length - 1];

        return sum >=0 ? 1 : 0;
    }

    void learn(double[] data, int realAnswer){
        if(data.length != this.weights.length - 1){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        int output = classify(data);
        int error = realAnswer - output;
        // aktualizacja wag
        for(int i = 0; i < this.weights.length - 1; i++){
            weights[i] += error * alphaLearningConstant * data[i];
        }

        // aktualizacja biasu
        weights[weights.length - 1] += error * alphaLearningConstant;

    }


    // Gettery
    public double[] getWeights() {
        return weights;
    }
    public double getAlphaLearningConstant(){
        return alphaLearningConstant;
    }
}
