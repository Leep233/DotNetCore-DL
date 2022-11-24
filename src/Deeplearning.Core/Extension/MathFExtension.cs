using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Extension
{
    public static class MathFExtension
    {
        public const double MIN_VALUE = 10E-15; //0.0000000001

        public static int Sum(params int[] args)
        {  
            if(args is null || args.Length<=0) return 0;

            int result = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                result += args[i];
            }
            return result;
        }
        public static double Sum(params double[] args)
        {
            if (args is null || args.Length <= 0) return 0;

            double result = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                result += args[i];
            }
            return result;
        }
        public static double Sum(params float[] args)
        {
            if (args is null || args.Length <= 0) return 0;

            double result = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                result += args[i];
            }
            return result;
        }
        public static double Sum(Vector args)
        {
            if (args.Length <= 0) return 0;

            double result = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                result += args[i];
            }
            return result;
        }
     

        public static double Average(params float[] args)
        {
            if (args is null || args.Length <= 0) return 0;

            return Sum(args) / args.Length;
        }
        public static double Average(params int[] args)
        {
            if (args is null || args.Length <= 0) return 0;

            return Sum(args) / args.Length;
        }
        public static double Average(Vector args)
        {
            if ( args.Length <= 0) return 0;

            return Sum(args) / args.Length;
        }
        public static double Average(params double[] args)
        {
            if (args.Length <= 0) return 0;

            return Sum(args) / args.Length;
        }


        public static int Min(params int[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            int value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value > args[i]) value = args[i];
            }
            return value;
        }
        public static float Min(params float[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            float value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value > args[i]) value = args[i];
            }
            return value;
        }
        public static double Min(params double[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            double value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value > args[i]) value = args[i];
            }
            return value;
        }
        public static double Min(Matrix matrix)
        {
            double result = matrix[0,0];
            for (int i = 1; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    if (result > matrix[i,j]) result = matrix[i,j];
                }               
            }
            return result;
        }
        public static double Min(Vector vector)
        {
            double value = vector[0];
            for (int i = 1; i < vector.Length; i++)
            {
                if (value > vector[i]) value = vector[i];
            }
            return value;
        }
      
        public static int Max(params int[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            int value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value < args[i]) value = args[i];
            }

            return value;
        }
        public static float Max(params float[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            float value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value < args[i]) value = args[i];
            }

            return value;
        }
        public static double Max(params double[] args)
        {
            if (args is null || args.Length <= 0)
                return 0;

            double value = args[0];

            for (int i = 1; i < args.Length; i++)
            {
                if (value < args[i]) value = args[i];
            }

            return value;
        }
        public static double Max(Vector vector)
        {
            double value = vector[0];
            for (int i = 1; i < vector.Length; i++)
            {
                if (value < vector[i]) value = vector[i];
            }
            return value;
        }
        public static double Max(Matrix matrix)
        {
            double result = matrix[0, 0];

            for (int i = 1; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    if (result < matrix[i, j]) result = matrix[i, j];
                }
            }
            return result;
        }


    }
}
