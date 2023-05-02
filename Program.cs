using System;
using System.IO;

namespace CsharpANN
{
    class Program
    {
        public static void Main()
        {
            using (var reader = new StreamReader(@"./mushrooms.csv"))
            {
                int[] networkShape = { 9, 18, 18, 2 };
                NeuralNetwork nn = new NeuralNetwork(networkShape, 1);
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

                    for (int i = 0; i < mushroom.GetInputNodes().Length; i++)
                    {
                        float[] hasil = nn.Process(mushroom.GetInputNodes());

                        Console.WriteLine("hasil");
                        Console.WriteLine(hasil[0]);
                        Console.WriteLine(hasil[1]);
                        Console.WriteLine("target");
                        Console.WriteLine(mushroom.EdibleValue());

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
            }

        }
    }

}