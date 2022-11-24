using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Deeplearning.Sample.Utils
{
    internal class ImageUtil
    {
        public static BitmapSource Pixels2Image(int width,int height, double dpiX, double dpiY,params byte[] colors)
        {

            int pixelWidth = width;

            int pixelHeight = height;

            int stride = pixelWidth * 4;

           return BitmapSource.Create(pixelWidth, pixelHeight, 
               dpiX, dpiY,
               PixelFormats.Bgr32,
               new BitmapPalette(new List<Color> { Colors.Blue , Colors.Green, Colors.Red}),
               colors, 
               stride);      
        }


        public static byte[] ReadImagePixels(BitmapImage img)
        {

            int stride = img.PixelWidth * 4;

            int size = img.PixelHeight * stride;

            byte[] pixels = new byte[size];

            img.CopyPixels(pixels, stride, 0);

            return pixels;

        }
    }
}
