using System;
using System.IO;

namespace CsharpANN
{
    class Program
    {
        public static void Main()
        {
            float max = 0;
            for (int i = 1; i <= 100; i++)
            {
                //Console.Write("=====================");
                //Console.WriteLine(100 * i);
                int maxTrain = 100 * i;
                float accuracy = TrainOnData(maxTrain);
                if (accuracy > max) max = accuracy;
                Console.WriteLine($"maxtrain: {maxTrain} accuracy: {accuracy}");
            }
            Console.WriteLine(max);
        }

        public static float TrainOnData(int maxTraining)
        {

            using (var reader = new StreamReader(@"./mushrooms.csv"))
            {
                int[] networkShape = { 9, 20, 2 };
                NeuralNetwork nn = new NeuralNetwork(networkShape, .01f, 100, "sigmoid");
                int numberOfTraining = 0;

                int numberOfTruth = 0;
                int numberOfFalse = 0;
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    Mushroom mushroom = new Mushroom(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9]
                    );

                    float[] hasil = nn.Process(mushroom.GetInputNodes());

                    if (numberOfTraining > maxTraining)
                    {
                        if (hasil[0] < hasil[1])
                        {
                            if (mushroom.EdibleValue() == 1f) numberOfTruth++;
                            else numberOfFalse++;
                        }
                        else
                        {
                            if (mushroom.EdibleValue() == 0f) numberOfTruth++;
                            else numberOfFalse++;
                        }
                    }

                    //training
                    numberOfTraining++;
                    if (numberOfTraining < maxTraining)
                    {
                        if (mushroom.EdibleValue() == 1)
                        {
                            float[] target = { 1, 0 };
                            nn.BackPropagate(target, mushroom.GetInputNodes());
                        }
                        else
                        {
                            float[] target = { 0, 1 };
                            nn.BackPropagate(target, mushroom.GetInputNodes());
                        }
                    }

                }

                //Console.WriteLine(numberOfTruth);
                //Console.WriteLine(numberOfFalse);
                float accuracy = numberOfTruth / ((float)numberOfTruth + (float)numberOfFalse);
                return accuracy;
            }
        }
    }
}
