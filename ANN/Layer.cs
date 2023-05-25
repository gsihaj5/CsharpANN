using System;

namespace CsharpANN
{

    public class Layer
    {
        public float[,] weightsArray;
        public float[,] deltaWeightsArray;
        public float[,] sumWeightError;
        public float[] biasesArray;
        public float[] deltaBiasArray;
        public float[] nodeArray;

        private int n_nodes;
        private int n_inputs;
        private int batch_index;
        private float learningRate;
        private string activation_function;

        public int GetInputCount()
        {
            return n_inputs;
        }
        public int GetNodeCount()
        {
            return n_nodes;
        }

        public Layer(int n_inputs, int n_nodes, float learningRate, int learning_batch, string activation_function)
        {
            this.n_inputs = n_inputs;
            this.n_nodes = n_nodes;
            this.learningRate = learningRate;
            this.activation_function = activation_function;
            this.batch_index = 0;

            weightsArray = new float[n_nodes, n_inputs];
            deltaWeightsArray = new float[n_nodes, n_inputs];
            Array.Clear(deltaWeightsArray, 0, deltaWeightsArray.Length);

            sumWeightError = new float[learning_batch, n_nodes];
            Array.Clear(sumWeightError, 0, sumWeightError.Length);

            biasesArray = new float[n_nodes];
            deltaBiasArray = new float[n_nodes];
            Array.Clear(deltaBiasArray, 0, deltaBiasArray.Length);

            nodeArray = new float[n_nodes];

            RandomNormal rand = new RandomNormal();
            double mean = 0.0;
            double standardDeviation = 5.0;

            //init bias
            for (int i = 0; i < biasesArray.Length; i++)
            {
                //biasesArray[i] = (float)rand.Next(-100, 100) / 100f;
                biasesArray[i] = (float) rand.NextNormal(mean,standardDeviation);
                //biasesArray[i] = 0;
            }

            //init weights
            for (int i = 0; i < n_nodes; i++)
            {
                for (int j = 0; j < n_inputs; j++)
                {
                    //weightsArray[i, j] = (float)rand.Next(-100, 100) / 100f;
                    weightsArray[i, j] = (float) rand.NextNormal(mean,standardDeviation);
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

        // Backward Propagation on Output layer
        public void BackwardOutput(float[] outputsArray, Layer prevLayer, int batchIndex)
        {
            float[] prevOutput = prevLayer.nodeArray;
            //weight adjustment
            for (int i = 0; i < n_nodes; i++)
            {
                float error = outputsArray[i] - nodeArray[i];
                float delI = -2 * error * ActivationDerivative(nodeArray[i]);
                deltaBiasArray[i] += delI;

                for (int j = 0; j < n_inputs; j++)
                {
                    deltaWeightsArray[i, j] += delI * prevOutput[j];
                    prevLayer.sumWeightError[batchIndex, j] += delI * weightsArray[i, j];
                }
            }
        }

        // Backward Propagation on Hidden layer
        public void BackwardHidden(Layer prevLayer, int batchIndex)
        {
            float[] prevOutput = prevLayer.nodeArray;
            //weight adjustment
            for (int i = 0; i < n_nodes; i++)
            {
                float delI = ActivationDerivative(nodeArray[i]) * sumWeightError[batchIndex, i];
                deltaBiasArray[i] += delI;

                for (int j = 0; j < n_inputs; j++)
                {
                    deltaWeightsArray[i, j] += delI * prevOutput[j];
                    prevLayer.sumWeightError[batchIndex, j] += delI * weightsArray[i, j];
                }
            }
        }

        // Backward Propagation on Input layer
        public void BackwardInput(float[] inputsArray, int batchIndex)
        {
            float[] prevOutput = inputsArray;
            //weight adjustment
            for (int i = 0; i < n_nodes; i++)
            {
                float delI = ActivationDerivative(nodeArray[i]) * sumWeightError[batchIndex, i];
                deltaBiasArray[i] += delI;

                for (int j = 0; j < n_inputs; j++)
                {
                    deltaWeightsArray[i, j] += delI * prevOutput[j];
                }
            }
        }

        public void UpdateWeight()
        {
            for (int i = 0; i < n_nodes; i++)
            {
                float deltaBias = deltaBiasArray[i] * learningRate;
                if (deltaBias > 0.00001f)
                    biasesArray[i] -= deltaBias;
                deltaBiasArray[i] = 0;
                for (int j = 0; j < n_inputs; j++)
                {
                    float deltaWeight = learningRate * deltaWeightsArray[i, j];
                    if (deltaWeight > 0.00001f)
                        weightsArray[i, j] -= deltaWeight;
                    deltaWeightsArray[i, j] = 0;
                }
            }
        }

        private float ActivationDerivative(float value){
            if(this.activation_function == "Relu") return ReLUDerivative(value);
            else if(this.activation_function == "Sigmoid") return SigmoidDerivative(value);

            return 0f;
        }

        private float ReLUDerivative(float value)
        {
            return value > 0 ? 1f : 0f;
        }

        private float SigmoidDerivative(float value)
        {
            return (1/(1+(MathF.Exp(-value))));
        }


        public void Activation()
        {
            if(this.activation_function == "Relu") ReluActivation();
            else if(this.activation_function == "Sigmoid") SigmoidActivation();
        }

        public void ReluActivation(){
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
