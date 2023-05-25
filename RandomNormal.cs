public class RandomNormal
{
    private Random random;
    private bool hasSpareValue;
    private double spareValue;

    public RandomNormal()
    {
        random = new Random();
        hasSpareValue = false;
        spareValue = 0.0;
    }

    public double NextNormal(double mean, double standardDeviation)
    {
        if (hasSpareValue)
        {
            hasSpareValue = false;
            return spareValue * standardDeviation + mean;
        }

        double u, v, s;
        do
        {
            u = 2.0 * random.NextDouble() - 1.0;
            v = 2.0 * random.NextDouble() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = Math.Sqrt(-2.0 * Math.Log(s) / s);
        spareValue = v * s;
        hasSpareValue = true;

        return mean + standardDeviation * u * s;
    }
}
