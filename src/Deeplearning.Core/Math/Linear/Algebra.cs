using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public static class Algebra
    {
        public static EigenEventArgs PowerIteration(Matrix matrix,int iterations = 300) 
        {
            double minValue = 1E-10;      

            int vectorCount = matrix.Column;

            Vector eigenVector = new Vector(vectorCount);//.Random(vectorCount, -3, 3);

            for (int i = 0; i < vectorCount; i++)
            {
                eigenVector[i] = new Random().NextDouble();
            }

            for (int i = 0; i < iterations; i++)
            {
                Vector  vector = matrix * eigenVector;

                double norm = Vector.Norm(vector);

                if (norm == 0)
                {
                    eigenVector = vector;
                    continue;
                }             

                vector /= norm;            

                Vector y = eigenVector - vector;

                norm = Vector.Norm(y);

                if (norm <= minValue) break;

                eigenVector = vector;
            }

            double[] eigenVector_T = eigenVector.T;

            double eigenValue = eigenVector_T * matrix * eigenVector;

            return new EigenEventArgs(eigenValue, eigenVector);

        }
   
    }
}
