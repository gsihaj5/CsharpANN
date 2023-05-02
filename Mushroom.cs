
namespace CsharpANN
{

    class Mushroom
    {
        private bool is_edible = false;
        private int cap_shape = 0;
        private int cap_surface = 0;
        private int cap_color = 0;
        private bool bruises = false;
        private int odor = 0;
        private int gill_attachment = 0;
        private int gill_spacing = 0;
        private int gill_size = 0;
        private int gill_color = 0;

        public Mushroom(
            string is_edible,
            string cap_shape,
            string cap_surface,
            string cap_color,
            string bruises,
            string odor,
            string gill_attachment,
            string gill_spacing,
            string gill_size,
            string gill_color
        )
        {
            this.is_edible = is_edible == "e";
            this.enumerateCapShape(cap_shape);
            this.enumerateCapSurface(cap_surface);
            this.enumerateCapColor(cap_color);
            this.bruises = bruises == "t";
            this.enumerateOdor(odor);
            this.enumerateGillAttachment(gill_attachment);
            this.enumerateGillSpacing(gill_spacing);
            this.enumerateGillSize(gill_size);
            this.enumerateGillColor(gill_color);

        }

        private void enumerateCapShape(string cap_shape)
        {
            switch (cap_shape)
            {
                case "b":
                    this.cap_shape = 0;
                    break;
                case "c":
                    this.cap_shape = 1;
                    break;
                case "x":
                    this.cap_shape = 2;
                    break;
                case "f":
                    this.cap_shape = 3;
                    break;
                case "k":
                    this.cap_shape = 4;
                    break;
                case "s":
                    this.cap_shape = 5;
                    break;
            }
        }

        private void enumerateCapSurface(string cap_surface)
        {
            switch (cap_surface)
            {
                case "f":
                    this.cap_surface = 0;
                    break;
                case "g":
                    this.cap_surface = 1;
                    break;
                case "y":
                    this.cap_surface = 2;
                    break;
                case "s":
                    this.cap_surface = 3;
                    break;
            }
        }

        private void enumerateCapColor(string cap_color)
        {
            switch (cap_color)
            {
                case "n":
                    this.cap_color = 0;
                    break;
                case "b":
                    this.cap_color = 1;
                    break;
                case "c":
                    this.cap_color = 2;
                    break;
                case "g":
                    this.cap_color = 3;
                    break;
                case "r":
                    this.cap_color = 4;
                    break;
                case "p":
                    this.cap_color = 5;
                    break;
                case "u":
                    this.cap_color = 6;
                    break;
                case "e":
                    this.cap_color = 7;
                    break;
                case "w":
                    this.cap_color = 8;
                    break;
                case "s":
                    this.cap_color = 9;
                    break;
            }
        }

        private void enumerateOdor(string odor)
        {
            switch (odor)
            {
                case "a":
                    this.odor = 0;
                    break;
                case "l":
                    this.odor = 1;
                    break;
                case "c":
                    this.odor = 2;
                    break;
                case "y":
                    this.odor = 3;
                    break;
                case "f":
                    this.odor = 4;
                    break;
                case "m":
                    this.odor = 5;
                    break;
                case "n":
                    this.odor = 6;
                    break;
                case "p":
                    this.odor = 7;
                    break;
                case "s":
                    this.odor = 8;
                    break;
            }
        }


        private void enumerateGillAttachment(string gill_attachment)
        {
            switch (gill_attachment)
            {
                case "a":
                    this.gill_attachment = 0;
                    break;
                case "d":
                    this.gill_attachment = 1;
                    break;
                case "f":
                    this.gill_attachment = 2;
                    break;
                case "n":
                    this.gill_attachment = 3;
                    break;
            }
        }

        private void enumerateGillSpacing(string gill_spacing)
        {
            switch (gill_spacing)
            {
                case "b":
                    this.gill_spacing = 0;
                    break;
                case "n":
                    this.gill_spacing = 1;
                    break;
            }
        }

        private void enumerateGillSize(string gill_size)
        {
            switch (gill_size)
            {
                case "b":
                    this.gill_size = 0;
                    break;
                case "n":
                    this.gill_size = 1;
                    break;
            }
        }

        private void enumerateGillColor(string gill_color)
        {
            switch (gill_color)
            {
                case "k":
                    this.gill_color = 0;
                    break;
                case "n":
                    this.gill_color = 1;
                    break;
                case "b":
                    this.gill_color = 2;
                    break;
                case "h":
                    this.gill_color = 3;
                    break;
                case "g":
                    this.gill_color = 4;
                    break;
                case "r":
                    this.gill_color = 5;
                    break;
                case "o":
                    this.gill_color = 6;
                    break;
                case "p":
                    this.gill_color = 7;
                    break;
                case "u":
                    this.gill_color = 8;
                    break;
                case "e":
                    this.gill_color = 9;
                    break;
                case "w":
                    this.gill_color = 10;
                    break;
                case "y":
                    this.gill_color = 11;
                    break;
            }
        }

        public float[] GetInputNodes()
        {
            float[] values = {
                (float)this.cap_shape,
                (float)this.cap_surface,
                (float)this.cap_color,
                (float)(this.bruises ? 1 : 0),
                (float)this.odor,
                (float)this.gill_attachment,
                (float)this.gill_spacing,
                (float)this.gill_size,
                (float)this.gill_color
            };
            return values;
        }
        public float EdibleValue()
        {
            return (float)(this.is_edible ? 1f : 0f);

        }
    }

}