
namespace CsharpANN
{
    class NeuralNetwork
    {
        public int[] networkShape;
        public Layer[] layers;
        private float learningRate;

        public NeuralNetwork(int[] networkShape, float learningRate)
        {
            this.learningRate = learningRate;
            this.networkShape = networkShape;

            layers = new Layer[networkShape.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(networkShape[i], networkShape[i + 1]);
            }
        }


        public float[] Process(float[] inputs)
        {
            for (int i = 0; i < layers.Length; i++)
            {
                if (i == 0)
                {
                    layers[i].Forward(inputs);
                    layers[i].Activation();
                }
                else if (i == layers.Length - 1)
                {
                    layers[i].Forward(layers[i - 1].nodeArray);
                    layers[i].Activation();
                }
            }

            //return the output value directly
            return layers[layers.Length - 1].nodeArray;
        }

        public void BackPropagate(float[] expected_output, float[] input_array)
        {
            float[][] errors = new float[layers.Length][];

            for (int i = layers.Length - 1; i >= 0; i--)
            {
                Layer current_layer = layers[i];

                if (i == layers.Length - 1) //output layer
                {
                    Layer prev_layer = layers[i - 1];


                    for (int node = 0; node < current_layer.nodeArray.Length; node++)
                    {
                        float delta = current_layer.nodeArray[node] - expected_output[node];

                        for (int input = 0; input < current_layer.GetInputCount(); input++)
                        {
                            z = current_layer.weightsArray[node, input] - prev_layer.nodeArray[input];
                            dLdZ = delta * ReLUDerivative(z);

                            errors[i] = 
                        }
                    }


                }

            }
        }
        private void ReLUDerivative(float value)
        {
            return value > 0 ? 1f : 0f;
        }

        public void UpdateWeight()
        {

        }
    }
}