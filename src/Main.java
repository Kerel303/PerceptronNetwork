import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Main {

    static boolean debugMode = false;
    static List<Iris> listToTeach = new ArrayList<>();
    static List<Iris> listToTest = new ArrayList<>();

    static Perceptron perceptron = new Perceptron(4, 0, 0.25);

    static int iterationNumber = 100;

    public static void main(String[] args) {
        getData();

        // Założenie: Iris setosa -> 1
        // Wszystkie pozostałe -> 0

        for(int i = 0; i < iterationNumber; i++){
            TeachPerceptron();
        }

        TestAccuracy();

    }




    
    static void TeachPerceptron(){
        for (Iris iris : listToTeach) {
            double[] data = {iris.sepalLength, iris.sepalWidth, iris.petalLength, iris.petalWidth};
            perceptron.learn(data, isSetosa(iris));
        }
    }

    static void TestAccuracy(){
        int counter = 0;
        for (Iris iris : listToTest) {
            double[] data = {iris.sepalLength, iris.sepalWidth, iris.petalLength, iris.petalWidth};

            if(perceptron.classify(data) == isSetosa(iris)){// Sprawdzamy, czy perceptron mówi prawdę
                counter++;
            }
        }
        double accuracy = ((double)counter / (double)listToTest.size()) * 100;
        System.out.println("Celność perceptronu: " + accuracy + "%");
    }


    static int isSetosa(Iris iris){
        if(iris.type.equals("Iris-setosa")){
            return 1;
        }else{
            return 0;
        }
    }
























    // Pobranie danych z pliku
    static void getData(){
        try(BufferedReader br = new BufferedReader(new FileReader("src/Iris.txt"))){
            String line;
            int counter = 1;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");

                Iris iris = new Iris(
                        Double.parseDouble(data[0]),
                        Double.parseDouble(data[1]),
                        Double.parseDouble(data[2]),
                        Double.parseDouble(data[3]),
                        data[4]
                );
                if(counter <= 3){
                    listToTeach.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Teach");
                    }
                    counter++;
                }else if(counter < 5){
                    listToTest.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Test");
                    }
                    counter++;
                }else if(counter == 5){
                    listToTest.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Test");
                    }
                    counter = 1;
                }

            }
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }
}
