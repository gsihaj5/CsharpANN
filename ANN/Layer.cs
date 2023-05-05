using System;

namespace CsharpANN
{

    public class Layer
    {
        public float[,] weightsArray;
        public float[,] deltaWeightsArray;
        public float[] biasesArray;
        public float[] deltaBiasArray;
        public float[] nodeArray;

        private int n_nodes;
        private int n_inputs;
        private int batch_number;
        private float learningRate;

        public int GetInputCount()
        {
            return n_inputs;
        }
        public int GetNodeCount()
        {
            return n_nodes;
        }

        //batch_number is how many training data grouped to update the weight
        public Layer(int n_inputs, int n_nodes, float learningRate)
        {
            this.n_inputs = n_inputs;
            this.n_nodes = n_nodes;
            this.learningRate = learningRate;

            weightsArray = new float[n_nodes, n_inputs];
            deltaWeightsArray = new float[n_nodes, n_inputs];
            Array.Clear(deltaWeightsArray, 0, deltaWeightsArray.Length);

            biasesArray = new float[n_nodes];
            deltaBiasArray = new float[n_nodes];
            Array.Clear(deltaBiasArray, 0, deltaBiasArray.Length);

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

        public void Backward(float[] outputsArray, float[] prevOutput)
        {
            //weight adjustment
            for (int i = 0; i < n_inputs; i++)
            {
                for (int j = 0; j < n_nodes; j++)
                {
                    deltaWeightsArray[j, i] += (error * ReLUDerivative(nodeArray[j])) * prevOutput[i];
                }
            }
        }

        private void ReLUDerivative(float value)
        {
            return value > 0 ? 1f : 0f;
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