
namespace CsharpANN
{
    class NeuralNetwork
    {
        public int[] networkShape;
        public Layer[] layers;
        public int learning_batch;
        public int learning_index;
        public float learning_rate;

        public NeuralNetwork(int[] networkShape, float learningRate, int learning_batch, string activation_function)
        {
            this.networkShape = networkShape;
            this.learning_batch = learning_batch;
            this.learning_index = 0;
            this.learning_rate = learningRate;

            layers = new Layer[networkShape.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(networkShape[i], networkShape[i + 1], learningRate, learning_batch, activation_function);
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

                    layers[i].BackwardOutput(expected_output, layers[i - 1], learning_index);
                }
                else if (i == 0)
                {
                    layers[i].BackwardInput(input_array, learning_index);
                }
                else
                {
                    layers[i].BackwardHidden(layers[i - 1], learning_index);
                }

            }

            learning_index++;

            if (learning_index == learning_batch)
            {
                UpdateWeight();
                learning_index = 0;
            }
        }

        public void UpdateWeight()
        {
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                Layer layer = layers[i];
                layer.UpdateWeight();
            }
        }
    }
}
