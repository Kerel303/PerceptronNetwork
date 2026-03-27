import java.util.ArrayList;
import java.util.List;

public class PerceptronLayer<T extends Trainable> {
    private List<Perceptron> perceptrons = new ArrayList<>();
    private List<String> classes = new ArrayList<>();
    private double activationThreshold = 0;
    private double alphaLearningConstant = 0.25; 
    private double betaLearningConstant = 0;

    PerceptronLayer(List<String> classes){
        this.classes = classes;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(classes.size(), this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, double activationThreshold){
        this.classes = classes;
        this.activationThreshold = activationThreshold;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(classes.size(), this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, double activationThreshold, double alphaLearningConstant){
        this.classes = classes;
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(classes.size(), this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }
    PerceptronLayer(List<String> classes, double activationThreshold, double alphaLearningConstant, double betaLearningConstant){
        this.classes = classes;
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
        this.betaLearningConstant = betaLearningConstant;
        for(int i = 0; i < classes.size(); i++){
            createPerceptron(classes.size(), this.activationThreshold, this.alphaLearningConstant, this.betaLearningConstant);
        }
    }

    void TeachPerceptrons(List<T> listToTeach){
        for (int i = 0; i < perceptrons.size(); i++){
            Perceptron p = perceptrons.get(i);
            String targetClass = classes.get(i);
            for (T t : listToTeach){
                int expected = t.getLabel().equals(targetClass) ? 1 : 0;
                p.learn(t.getInput(), expected);
            }
        }
    }


    
    public void addPerceptron(Perceptron perceptron){
        perceptrons.add(perceptron);
    }
    public void createPerceptron(int LengthOfTheWeightVector, double activationThreshold, double alphaLearningConstant){
        Perceptron perceptron = new Perceptron(LengthOfTheWeightVector, activationThreshold, alphaLearningConstant);
        perceptrons.add(perceptron);
    }
    public void createPerceptron(int LengthOfTheWeightVector, double activationThreshold, double alphaLearningConstant, double betaLearningConstant){
        Perceptron perceptron = new Perceptron(LengthOfTheWeightVector, activationThreshold, alphaLearningConstant, betaLearningConstant);
        perceptrons.add(perceptron);
    }
}
