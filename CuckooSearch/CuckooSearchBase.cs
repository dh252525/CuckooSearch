using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CuckooSearch
{
    public class CuckooSearchBase
    {
        public double[,] GetNewNestVialevy(double[,] xt, double[] xBest, int[] lb, int[] ub, double lamuda)
        {

            var xtTarget = DenseMatrix.OfArray(xt);
            double beta = 1.5;

            double sigmau = Math.Pow(SpecialFunctions.Gamma(1 + beta) * Math.Sin(Math.PI * beta / 2) / (
                SpecialFunctions.Gamma((1 + beta) / 2) * beta * Math.Pow(2, (beta - 1) / 2)), (1 / beta));

            double sigmav = 1;

            for (int i = 0; i < xt.GetLength(0); i++)
            {
                //取多维数组某行元素
                var s = DimensionalityReduction(xt, i);
                /*mean-stdev-size*/
                /*u = np.random.normal(0, sigma_u, 1)*/
                double[] u = Generate.Normal(1, 0, sigmau);
                double[] v = Generate.Normal(1, 0, sigmav);

                double ls = u[0] / Math.Pow(Math.Abs(v[0]), (1 / beta));

                //lamuda的设置关系到点的活力程度  方向是由最佳位置确定的  有点类似PSO算法  但是步长不一样
                double[] e = new double[s.Length];
                for (int j = 0; j < s.Length; j++)
                {
                    e[j] = s[j] - xBest[j];
                }

                var stepsize = e.Select(x => x * lamuda * ls).ToArray();
                //产生满足正态分布的序列
                Random r = new Random();
                for (int k = 0; k < s.Length; k++)
                {
                    s[k] = s[k] + stepsize[k] * r.NextDouble();
                }
               
                //重新给数组某行赋值
                var xtNew = DenseMatrix.OfArray(xt);
                xtNew.SetRow(i, s);
                var sBounds = SimpleBounds(s, lb, ub);
                xtNew.SetRow(i, sBounds);
                xtTarget = xtNew;

            }
            return xtTarget.ToArray();
        }

        /// <summary>
        /// 约束迭代结果一维
        /// </summary>
        /// <param name="s"></param>
        /// <param name="lb"></param>
        /// <param name="ub"></param>
        /// <returns></returns>
        public double[] SimpleBounds(double[] s, int[] lb, int[] ub) {
            double[] arrayResult = new double[s.Length];
            for (int i = 0; i < s.GetLength(0); i++)
            {
                if (s[i] < lb[i])
                {
                    arrayResult[i] = lb[i];
                }
                if (s[i] > ub[i])
                {
                    arrayResult[i] = ub[i];
                }

            }
            return arrayResult;
        }

        /// <summary>
        /// 约束迭代结果二维
        /// </summary>
        /// <param name="s"></param>
        /// <param name="lb"></param>
        /// <param name="ub"></param>
        /// <returns></returns>
        public double[,] DimensionalSimpleBounds(double[,] s, int[] lb, int[] ub)
        {
            double[,] arrayResult = new double[s.GetLength(0), s.GetLength(1)];
            Array.Copy(s, arrayResult, s.Length);
            for (int i = 0; i < arrayResult.GetLength(0); i++)
            {
                for (int j = 0; j < arrayResult.GetLength(1); j++)
                {
                    if (arrayResult[i, j] < lb[j])
                    {
                        arrayResult[i, j] = lb[j];
                    }
                    if (arrayResult[i, j] > ub[j])
                    {
                        arrayResult[i, j] = ub[j];
                    }
                }
            }
            return arrayResult;
        }

        /// <summary>
        /// 获得当前最优解
        /// </summary>
        /// <param name="cukoo"></param>
        /// <returns></returns>
        public Cukoo GetBestNest(Cukoo cukoo) {
            double fitall = 0;
            for (int i = 0; i < cukoo.Nest.GetLength(0); i++)
            {
                double temp1 = FitNess(DimensionalityReduction(cukoo.Nest, i));
                double temp2 = FitNess(DimensionalityReduction(cukoo.newNest, i));
                if (temp1 > temp2)
                {
                    //old origin 无必要修改
                    cukoo.Nest = SetupDimensionalArray(cukoo.Nest, cukoo.newNest, i);
                    if (temp2 < cukoo.NBest)
                    {
                        cukoo.NBest = temp2;
                        cukoo.NestBest = SetupArray(cukoo.NestBest, cukoo.Nest, i);
                        fitall = fitall + temp2;
                    }
                }
                else
                {
                    fitall = fitall + temp1;
                }
                var meanfit = DimensionalityReduction(cukoo.Nest, 0);
                cukoo.NestBest = meanfit.Select(x => (fitall / x)).ToArray();

            }
            return cukoo;
        }

        

        /// <summary>
        /// 适应度计算
        /// </summary>
        /// <param name="nestN"></param>
        /// <returns></returns>
        public double FitNess(double[] nestN)
        {
            double x = nestN[0];
            double y = nestN[1];
            //rastrigin函数
            double a = 10;
            double z = 2 * a + Math.Pow(x, 2) - a * Math.Cos(2 * Math.PI * x) + Math.Pow(y, 2);
            return z;
        }
        /// <summary>
        /// 按pa抛弃部分巢穴
        /// </summary>
        /// <param name="nest"></param>
        /// <param name="lb"></param>
        /// <param name="ub"></param>
        /// <param name="pa"></param>
        /// <returns></returns>
        public double[,] EmptyNest(double[,] nest, int[] lb, int[] ub, double pa)
        {
            double n = nest.GetLength(0);
            double[,] nest1 = new double[nest.GetLength(0), nest.GetLength(1)];
            double[,] nest2 = new double[nest.GetLength(0), nest.GetLength(1)];
            Array.Copy(nest, nest1, nest.Length);
            Array.Copy(nest, nest2, nest.Length);
            var randM = GetRandomDimensionalArray(nest);
            randM = Heaviside(randM, 0);
            Shuffle(ref nest1);
            Shuffle(ref nest2);

            double[,] arr = new double[1, 1];

            //补充数组的行和列再运算
            var rand = GetRandomDimensionalArray(arr);
            var m1 = rand[0, 0];
            var m2 = DenseMatrix.OfArray(ArraySubtrac(nest1, nest2));
            var stepSize = m1 * m2;
            var newStep = ArrayMultiply(stepSize.ToArray(), randM);
            var newNest = DenseMatrix.OfArray(nest) + DenseMatrix.OfArray(newStep);
            var nestReturn = DimensionalSimpleBounds(newNest.ToArray(), lb, ub);
            return nestReturn;

        }


        /// <summary>
        /// 布谷鸟算法
        /// </summary>
        /// <param name="lamuda"></param>
        /// <param name="pa"></param>
        /// <returns></returns>
        public double GetCs(double lamuda = 1, double pa = 0.25) {
            int[] lb = { -5, -5 };
            int[] ub = { 5, 5 };
            int populationSize = 20;
            int dim = 2;
            double[,] arrTemp = new double[populationSize, dim];

            double[,] nest = GetRandomDimensionalArray(arrTemp, lb[0], ub[0]);
            double[] nestBest = DimensionalityReduction(nest, 0);
            double nBest = FitNess(nestBest);
            Cukoo cukoo = new Cukoo()
            {
                Nest = nest,
                NBest = nBest,
                NestBest = nestBest,
                newNest = nest
            };
            cukoo = GetBestNest(cukoo);
            for (int i = 0; i < 30; i++)
            {
                double[,] nestC = new double[cukoo.Nest.GetLength(0), cukoo.Nest.GetLength(1)];

                Array.Copy(cukoo.Nest,nestC,cukoo.Nest.Length);
                //根据莱维飞行产生新的位置
                cukoo.newNest = GetNewNestVialevy(nestC, cukoo.NestBest, lb, ub, lamuda);
                //判断新的位置优劣进行替换
                cukoo = GetBestNest(cukoo);

                double[,] nestE = new double[cukoo.Nest.GetLength(0), cukoo.Nest.GetLength(1)];

                Array.Copy(cukoo.Nest, nestE, cukoo.Nest.Length);
                //丢弃部分巢穴
                cukoo.newNest = EmptyNest(nestE, lb, ub, pa);
                cukoo = GetBestNest(cukoo);
                
            }
            //最优解为 cukoo.NBest
            return cukoo.NBest;

        }

        /// <summary>
        /// 洗牌算法
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        public void Shuffle<T>(ref T[,] list)
        {
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            T[,] newArray = new T[list.GetLength(0), list.GetLength(1)];
            List<T> newList = new List<T>();//储存结果的集合
            foreach (T item in list)
            {
                newList.Insert(rand.Next(0, newList.Count), item);
            }
            newList.Remove(list[0, 0]);//移除list[0]的值
            newList.Insert(rand.Next(0, newList.Count), list[0, 0]);//再重新随机插入第一笔
            for (int i = 0; i < list.GetLength(0); i++)
            {
                for (int j = 0; j < list.GetLength(1); j++)
                {
                    newArray[i, j] = newList[i + j];
                }
            }
            list = newArray;
        }

        /// <summary>
        /// 海维赛德阶跃函数
        /// </summary>
        /// <param name="x1">输入值</param>
        /// <param name="x2">函数值</param>
        /// <returns></returns>
        public double[,] Heaviside(double[,] x1, double x2)
        {
            double[,] newX1 = new double[x1.GetLength(0), x1.GetLength(1)];
            for (int i = 0; i < x1.GetLength(0); i++)
            {

                for (int j = 0; j < x1.GetLength(1); j++)
                {
                    var v = x1[i, j];
                    if (v < 0)
                    {
                        newX1[i, j] = 0;
                    }
                    else if (v == 0)
                    {
                        newX1[i, j] = x2;
                    }
                    else if (v > 0)
                    {
                        newX1[i, j] = 1;
                    }
                }
            }
            return newX1;
        }

        /// <summary>
        /// 生成随机二维数组
        /// </summary>
        /// <param name="num"></param>
        /// <param name="minValue"></param>
        /// <param name="maxValue"></param>
        /// <returns></returns>
        public double[,] GetRandomDimensionalArray(double[,] num)
        {
            //new math.net
            var mbArr = Matrix<double>.Build;
            var randomMatrix = mbArr.Random(num.GetLength(0), num.GetLength(1));
            return randomMatrix.ToArray();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="num"></param>
        /// <param name="minValue"></param>
        /// <param name="maxValue"></param>
        /// <returns></returns>
        public T[,] GetRandomDimensionalArray<T>(T[,] num,int minValue,int maxValue)
        {
            T[,] a = new T[num.GetLength(0), num.GetLength(1)];
            Random r = new Random();
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    var time = r.Next(minValue, maxValue);
                    a[i, j] = (T)Convert.ChangeType(r.NextDouble() * time, typeof(T));
                }
            }
            return a;
        }

        /// <summary>
        /// 生成随机二维数组包括减数
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="num"></param>
        /// <param name="minValue"></param>
        /// <param name="maxValue"></param>
        /// <param name="subtractor"></param>
        /// <returns></returns>
        public T[,] GetRandomDimensionalArray<T>(T[,] num, int minValue, int maxValue,int subtractor)
        {
            T[,] a = new T[num.GetLength(0), num.GetLength(1)];
            Random r = new Random();
            
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    if (subtractor == 0)
                    {
                        a[i, j] = (T)Convert.ChangeType(r.Next(minValue, maxValue), typeof(T));
                    }
                    else
                    {
                        a[i, j] = (T)Convert.ChangeType(subtractor - r.Next(minValue, maxValue), typeof(T));
                    }
                }
            }
            return a;
        }
        /// <summary>
        /// 生成随机一维数组
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="num"></param>
        /// <param name="minValue"></param>
        /// <param name="maxValue"></param>
        /// <returns></returns>
        public T[] GetRandomArray<T>(T[] num, int minValue, int maxValue)
        {
            T[] a = new T[num.Length];
            Random r = new Random();
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = (T)Convert.ChangeType(r.Next(minValue, maxValue), typeof(T));
            }
            return a;
        }

        /// <summary>
        /// 获取二维数组某行元素
        /// </summary>
        /// <param name="num"></param>
        /// <param name="row">行数</param>
        /// <returns></returns>

        public double[] DimensionalityReduction(double[,] num, int lineRow)
        {
            //new math.net
            var arr = DenseMatrix.OfArray(num).Row(lineRow);
            return arr.ToArray();
          
        }

        /// <summary>
        /// 二维给二维数组赋值
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array">目标数组</param>
        /// <param name="values">复制数组</param>
        /// <param name="lineNum"></param>
        /// <returns></returns>
        public double[,] SetupDimensionalArray(double[,] destinationArray, double[,] sourceArray, int lineNum)
        {
            double[,] resultArray = new double[destinationArray.GetLength(0), destinationArray.GetLength(1)];
            //new math.net
            var desArray = DenseMatrix.OfArray(destinationArray);
            var souArray = DenseMatrix.OfArray(sourceArray);
            var rowArray = souArray.Row(lineNum);
            desArray.SetRow(lineNum, rowArray);
            resultArray = desArray.ToArray();
            return resultArray;
            //old origin
            //for (int i = 0; i < destinationArray.GetLength(1); i++)
            //{
            //    resultArray[lineNum, i] = sourceArray[lineNum, i];
            //}
            //return resultArray;
        }

        /// <summary>
        /// 二维给一维数组赋值
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array">目标数组</param>
        /// <param name="values">复制数组</param>
        /// <param name="lineNum"></param>
        /// <returns></returns>
        public T[] SetupArray<T>(T[] array, T[,] values, int lineNum)
        {
            T[] a = new T[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = values[lineNum, i];
            }
            return a;
        }
        /// <summary>
        /// 一维给二维数组赋值
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="values"></param>
        /// <param name="lineNum"></param>
        /// <returns></returns>
        public T[,] ArraySetupDimensional<T>(T[,] array, T[] values, int lineNum)
        {
            T[,] a = new T[array.GetLength(0), array.GetLength(1)];
            for (int i = 0; i < array.GetLength(1); i++)
            {
                array[lineNum, i] = values[i];
            }
            return a;
        }

        /// <summary>
        /// 数组相减
        /// </summary>
        /// <param name="array"></param>
        /// <param name="array1"></param>
        /// <returns></returns>
        public double[,] ArraySubtrac(double[,] array, double[,] array1)
        {
            //new math.net
            var arrayD = DenseMatrix.OfArray(array);
            var arrayD1 = DenseMatrix.OfArray(array1);
            var arrayResult = arrayD - arrayD1;
            return arrayResult.ToArray();
        }
        /// <summary>
        /// 数组相乘
        /// </summary>
        /// <param name="array"></param>
        /// <param name="array1"></param>
        /// <returns></returns>
        public double[,] ArrayMultiply(double[,] array, double[,] array1)
        {
            if ((array.GetLength(0) != array1.GetLength(0)) || (array.GetLength(1) != array1.GetLength(1)))
            {
                throw new Exception("Matrix dimensions must agree");
            }
            double[,] arrayResult = new double[array.GetLength(0), array.GetLength(1)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    arrayResult[i, j] = array[i, j] * array1[i, j];
                }
            }
            return arrayResult;
        }
    }

    public class Cukoo
    {
        public double[,] Nest
        {
            get; set;
        }

        public double[,] newNest
        {
            get; set;
        }

        public double NBest
        {
            get; set;
        }

        public double[] NestBest
        {
            get; set;
        }
    }
}
