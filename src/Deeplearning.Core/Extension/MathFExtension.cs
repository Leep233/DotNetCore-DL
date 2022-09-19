using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Extension
{
    public static class MathFExtension
    {
        public static double Sum(int[] array)
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                sum += array[i];
            }
            return sum;
        }
        public static double Sum(double[] array)
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                sum += array[i];
            }
            return sum;
        }
        public static double Sum(Vector array)
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                sum += array[i];
            }
            return sum;
        }
        public static double Sum(float[] array) 
        {
            double sum = 0;

            for (int i = 0; i < array.Length; i++)
            {
                sum += array[i];
            }
            return sum;
        }

        public static double Average(float[] array)
        {
            int length = array.Length;

            if (length <= 0) return 0;

            double sum = Sum(array);
 
            return sum / length;
        }
        public static double Average(int[] array)
        {
            int length = array.Length;

            if (length <= 0) return 0;

            double sum = Sum(array);

            return sum / length;
        }
        public static double Average(Vector array)
        {
            int length = array.Length;

            if (length <= 0) return 0;

            double sum = Sum(array);

            return sum / length;
        }
        public static double Average(double[] array)
        {

            int length = array.Length;

            if (length <= 0) return 0;

            double sum = Sum(array);

            return sum / length;
        }


        public static double Min(Matrix v)
        {
            double value = v[0,0];
            for (int i = 1; i < v.Row; i++)
            {
                for (int j = 0; j < v.Column; j++)
                {
                    if (value > v[i,j]) value = v[i,j];
                }               
            }
            return value;
        }
        public static double Max(Matrix v)
        {
            double value = v[0, 0];
            for (int i = 1; i < v.Row; i++)
            {
                for (int j = 0; j < v.Column; j++)
                {
                    if (value < v[i, j]) value = v[i, j];
                }
            }
            return value;
        }
        public static double Min(double[] v)
        {
            double value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value > v[i]) value = v[i];
            }
            return value;
        }
        public static double Max(double[] v)
        {
            double value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value < v[i]) value = v[i];
            }
            return value;
        }
        public static double Min(Vector v)
        {
            double value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value > v[i]) value = v[i];
            }
            return value;
        }
        public static double Max(Vector v)
        {
            double value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value < v[i]) value = v[i];
            }
            return value;
        }

    }
}
