package com.ivan.learning.lesson1;

import java.util.Map;

public class PolynomialSecondDegree {
    private static final Map<Float, Float> functionValues = Map.of(
            -0.15f, 0.6151f, -0.10f, 0.6418f, -0.05f, 0.6678f,
            0.00f, 0.6931f, 0.05f, 0.7178f, 0.10f, 0.7419f, 0.15f, 0.7654f);

    public static float[] leastSquaresQuadratic() {
        int n = functionValues.size();
        float sumX = 0, sumY = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0, sumXY = 0, sumX2Y = 0;

        for (Map.Entry<Float, Float> entry : functionValues.entrySet()) {
            float x = entry.getKey();
            float y = entry.getValue();
            sumX += x;
            sumY += y;
            sumX2 += x * x;
            sumX3 += x * x * x;
            sumX4 += x * x * x * x;
            sumXY += x * y;
            sumX2Y += x * x * y;
        }

        float[][] matrix = {
                {n,     sumX,  sumX2}, // c, b, a
                {sumX,  sumX2, sumX3},
                {sumX2, sumX3, sumX4}
        };
        float[] constants = {sumY, sumXY, sumX2Y};

        return solveSystem(matrix, constants); // [c, b, a]
    }

    private static float[] solveSystem(float[][] matrix, float[] constants) {
        int n = constants.length;
        float[] result = new float[n];

        // Прямой ход метода Гаусса
        for (int i = 0; i < n; i++) {
            float pivot = matrix[i][i];
            if (pivot == 0) throw new ArithmeticException("Деление на ноль в методе Гаусса");

            for (int j = i; j < n; j++) {
                matrix[i][j] /= pivot;
            }
            constants[i] /= pivot;

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    float factor = matrix[k][i];
                    for (int j = i; j < n; j++) {
                        matrix[k][j] -= factor * matrix[i][j];
                    }
                    constants[k] -= factor * constants[i];
                }
            }
        }

        System.arraycopy(constants, 0, result, 0, n);

        return result; // [c, b, a]
    }

    public static void main(String[] args) {
        float[] coeffs = leastSquaresQuadratic();
        // coeffs[0] = c, coeffs[1] = b, coeffs[2] = a
        System.out.printf("f(x) = %.4f * x² + %.4f * x + %.4f%n", coeffs[2], coeffs[1], coeffs[0]);

        for (float x : functionValues.keySet()) {
            float y = coeffs[2] * x * x + coeffs[1] * x + coeffs[0];
            System.out.printf("x = %.2f, f(x) = %.4f, actual = %.4f%n", x, y, functionValues.get(x));
        }
    }
}
// y = -0.1271 * x² + 0.5008 * x + 0.6931
// https://www.wolframalpha.com/input?i=plot+-0.1271x^2+%2B+0.5008x+%2B+0.6931
