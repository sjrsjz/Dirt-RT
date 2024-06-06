#ifndef _HDR_
#define _HDR_
//#define array_size 13
/*void linear_HDR_AB_N(in float X_n[array_size], out float A,out float B){
    //first we sort the array, from smallest to largest
    for(int i = 0; i < array_size; i++){
        for(int j = i+1; j < array_size; j++){
            if(X_n[i] > X_n[j]){
                float temp = X_n[i];
                X_n[i] = X_n[j];
                X_n[j] = temp;
            }
        }
    }
    //then we calculate sum of X_n, x_n^2, x_n*n, x_n^2*n,x_n^2*n^2
    float sum_X_n = 0.0;
    float sum_X_n2 = 0.0;
    float sum_X_n_n = 0.0;
    float sum_X_n2_n = 0.0;
    float sum_X_n2_n2 = 0.0;
    for(int i = 0; i < array_size; i++){
        sum_X_n += X_n[i];
        sum_X_n2 += X_n[i]*X_n[i];
        sum_X_n_n += X_n[i]*i;
        sum_X_n2_n += X_n[i]*X_n[i]*i;
        sum_X_n2_n2 += X_n[i]*X_n[i]*i*i;
    }
    //then we calculate A and B
    float det = sum_X_n2_n2*sum_X_n2 - sum_X_n2_n*sum_X_n2_n;
    A = (sum_X_n2*sum_X_n_n - sum_X_n2_n*sum_X_n)/det;
    B = (sum_X_n2_n2*sum_X_n - sum_X_n2_n*sum_X_n_n)/det;
}

float linear_HDR_N(float x, float A, float B){
    return A + B*x;
}
*/

void linear_HDR_AB(float sum_X_n,float sum_X_n2,float sum_X_n3,float sum_X_n4, out float A,out float B){
    float det = sum_X_n4*sum_X_n2 - sum_X_n3*sum_X_n3;
    A = (sum_X_n2*sum_X_n2 - sum_X_n3*sum_X_n)/det;
    B = (sum_X_n4*sum_X_n - sum_X_n3*sum_X_n2)/det;
}

float linear_HDR(float x, float A, float B){
    return A*x + B;
}


void div_HDR_AB(float sum_X,float sum_X2,float sum_div,float sum_div_2,float sum_div2, out float A,out float B){
    float det = sum_div_2*sum_div_2-sum_div2*sum_X2;
    A=(sum_div_2*sum_X-sum_div*sum_X2)/det;
    B=(sum_div_2*sum_div-sum_X*sum_div2)/det;
}


float div_HDR(float x, float A, float B,float C){
    return A/(x+C) + B;
}

#endif