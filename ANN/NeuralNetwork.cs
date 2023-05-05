
namespace CsharpANN
{
    class NeuralNetwork
    {
        public int[] networkShape;
        public Layer[] layers;

        public NeuralNetwork(int[] networkShape, float learningRate)
        {
            this.networkShape = networkShape;

            layers = new Layer[networkShape.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(networkShape[i], networkShape[i + 1], learningRate);
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
                else
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
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                if (i == layers.Length - 1) //1 layer before output
                {

                    layers[i].Backward(expected_output, layers[i - 1].nodeArray);
                }
                else
                {
                    layers[i].Backward(layers[i + 1]);
                }
            }
        }

        public void UpdateWeight()
        {

        }
    }
}