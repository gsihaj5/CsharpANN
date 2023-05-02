using System;

namespace CsharpANN
{

    public class Layer
    {
        public float[,] weightsArray;
        public float[] biasesArray;
        public float[] nodeArray;

        private int n_nodes;
        private int n_inputs;
        private int batch_number;

        public int GetInputCount()
        {
            return n_inputs;
        }
        public int GetNodeCount()
        {
            return n_nodes;
        }

        //batch_number is how many training data grouped to update the weight
        public Layer(int n_inputs, int n_nodes, int batch_number)
        {
            this.n_inputs = n_inputs;
            this.n_nodes = n_nodes;
            this.batch_number = batch_number;

            weightsArray = new float[n_nodes, n_inputs];
            biasesArray = new float[n_nodes];
            nodeArray = new float[n_nodes];

            Random rand = new Random();

            //init bias
            for (int i = 0; i < biasesArray.Length; i++)
            {
                biasesArray[i] = (float)rand.Next(-100, 100) / 100f;
            }

            //init weights
            for (int i = 0; i < n_nodes; i++)
            {
                for (int j = 0; j < n_inputs; j++)
                {
                    weightsArray[i, j] = (float)rand.Next(-100, 100) / 100f;
                }
            }
        }

        public void Forward(float[] inputsArray)
        {
            nodeArray = new float[n_nodes];

            for (int i = 0; i < n_nodes; i++)
            {
                for (int j = 0; j < n_inputs; j++)
                {
                    nodeArray[i] += weightsArray[i, j] * inputsArray[j];
                }
                nodeArray[i] += biasesArray[i];
            }
        }

        public void Activation()
        {
            for (int i = 0; i < n_nodes; i++)
            {
                //relu
                if (nodeArray[i] < 0)
                {
                    nodeArray[i] = 0;
                }
            }
        }
        public void SigmoidActivation()
        {
            for (int i = 0; i < n_nodes; i++)
            {
                nodeArray[i] = 1 / (float)(1 + Math.Pow(Math.E, -nodeArray[i]));
            }
        }
    }
}