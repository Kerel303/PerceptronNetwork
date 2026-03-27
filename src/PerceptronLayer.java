import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PerceptronLayer<T extends Trainable> {
    private List<Perceptron> perceptrons = new ArrayList<>();
    private List<String> classes = new ArrayList<>();
    private int inputSize = 0;
    private double activationThreshold = 0;
    private double alphaLearningConstant = 0.25; 
    private double betaLearningConstant = 0;

    // Konstruktory
    PerceptronLayer(List<String> classes, int inputSize){
        this.classes = classes;
        this.inputSize = inputSize;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(this.inputSize, this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, int inputSize, double activationThreshold){
        this.classes = classes;
        this.inputSize = inputSize;
        this.activationThreshold = activationThreshold;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(this.inputSize, this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, int inputSize, double activationThreshold, double alphaLearningConstant){
        this.classes = classes;
        this.inputSize = inputSize;
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(this.inputSize, this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, int inputSize, double activationThreshold, double alphaLearningConstant, double betaLearningConstant){
        this.classes = classes;
        this.inputSize = inputSize;
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
        this.betaLearningConstant = betaLearningConstant;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(this.inputSize, this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }

    // Metody

    // Uczenie perceptronów
    public void train(List<T> data, int epochs){
        for(int e = 0; e < epochs; e++){
            TeachPerceptrons(data);
        }
    }

    private void TeachPerceptrons(List<T> listToTeach){
        for (int i = 0; i < perceptrons.size(); i++){
            Collections.shuffle(listToTeach);
            Perceptron p = perceptrons.get(i);
            String targetClass = classes.get(i);
            for (T t : listToTeach){
                int expected = t.getLabel().equals(targetClass) ? 1 : 0;
                p.learn(t.getInput(), expected);
            }
        }
    }

    // Klasyfikacja perceptronów
    public String classify(T t){
        double maxScore = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;

        for(int i = 0; i < perceptrons.size(); i++){
            Perceptron p = perceptrons.get(i);
            double sum = 0;
            double[] data = t.getInput();
            double[] weights = p.getWeights();
            for(int j = 0; j < weights.length; j++){
                sum += data[j] * weights[j];
            }
            if(sum > maxScore){
                maxScore = sum;
                bestIndex = i;
            }
        }
        if(bestIndex == -1){
            throw new IllegalStateException("Brak klasyfikacji - sprawdź dane");
        }
        return classes.get(bestIndex);
    }

    // Test celności warstwy perceeptronów
    public double accuracy(List<T> testData){
        int correct = 0;
        for(T t : testData){
            if(classify(t).equals(t.getLabel())){
                correct++;
            }
        }
        return (double) correct / testData.size();
    }


    // Tworzenie perceptronu
    private void createPerceptron(int LengthOfTheWeightVector, double activationThreshold, double alphaLearningConstant, double betaLearningConstant){
        Perceptron perceptron = new Perceptron(LengthOfTheWeightVector, activationThreshold, alphaLearningConstant, betaLearningConstant);
        perceptrons.add(perceptron);
    }
}
