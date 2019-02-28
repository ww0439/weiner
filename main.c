//维纳滤波，连续频谱最小值跟踪

#define DR_WAV_IMPLEMENTATION
#include"dr_wav.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"fftw3.h"
#pragma comment(lib,"C:\\Users\\Administrator\\Desktop\\wiener\\libfftw3-3.lib")
#pragma comment(lib,"C:\\Users\\Administrator\\Desktop\\wiener\\libfftw3f-3.lib")
#pragma comment(lib,"C:\\Users\\Administrator\\Desktop\\wiener\\libfftw3l-3.lib")



#define DD_PR_SNR     (float)0.7
#define framesize      (int)256

static const float win256[256] = {
        (float) 0.00000000, (float) 0.01636173, (float) 0.03271908, (float) 0.04906767, (float) 0.06540313,
        (float) 0.08172107, (float) 0.09801714, (float) 0.11428696, (float) 0.13052619, (float) 0.14673047,
        (float) 0.16289547, (float) 0.17901686, (float) 0.19509032, (float) 0.21111155, (float) 0.22707626,
        (float) 0.24298018, (float) 0.25881905, (float) 0.27458862, (float) 0.29028468, (float) 0.30590302,
        (float) 0.32143947, (float) 0.33688985, (float) 0.35225005, (float) 0.36751594, (float) 0.38268343,
        (float) 0.39774847, (float) 0.41270703, (float) 0.42755509, (float) 0.44228869, (float) 0.45690388,
        (float) 0.47139674, (float) 0.48576339, (float) 0.50000000, (float) 0.51410274, (float) 0.52806785,
        (float) 0.54189158, (float) 0.55557023, (float) 0.56910015, (float) 0.58247770, (float) 0.59569930,
        (float) 0.60876143, (float) 0.62166057, (float) 0.63439328, (float) 0.64695615, (float) 0.65934582,
        (float) 0.67155895, (float) 0.68359230, (float) 0.69544264, (float) 0.70710678, (float) 0.71858162,
        (float) 0.72986407, (float) 0.74095113, (float) 0.75183981, (float) 0.76252720, (float) 0.77301045,
        (float) 0.78328675, (float) 0.79335334, (float) 0.80320753, (float) 0.81284668, (float) 0.82226822,
        (float) 0.83146961, (float) 0.84044840, (float) 0.84920218, (float) 0.85772861, (float) 0.86602540,
        (float) 0.87409034, (float) 0.88192126, (float) 0.88951608, (float) 0.89687274, (float) 0.90398929,
        (float) 0.91086382, (float) 0.91749450, (float) 0.92387953, (float) 0.93001722, (float) 0.93590593,
        (float) 0.94154407, (float) 0.94693013, (float) 0.95206268, (float) 0.95694034, (float) 0.96156180,
        (float) 0.96592583, (float) 0.97003125, (float) 0.97387698, (float) 0.97746197, (float) 0.98078528,
        (float) 0.98384601, (float) 0.98664333, (float) 0.98917651, (float) 0.99144486, (float) 0.99344778,
        (float) 0.99518473, (float) 0.99665524, (float) 0.99785892, (float) 0.99879546, (float) 0.99946459,
        (float) 0.99986614, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000, (float) 1.00000000,
        (float) 1.00000000, (float) 0.99986614, (float) 0.99946459, (float) 0.99879546, (float) 0.99785892,
        (float) 0.99665524, (float) 0.99518473, (float) 0.99344778, (float) 0.99144486, (float) 0.98917651,
        (float) 0.98664333, (float) 0.98384601, (float) 0.98078528, (float) 0.97746197, (float) 0.97387698,
        (float) 0.97003125, (float) 0.96592583, (float) 0.96156180, (float) 0.95694034, (float) 0.95206268,
        (float) 0.94693013, (float) 0.94154407, (float) 0.93590593, (float) 0.93001722, (float) 0.92387953,
        (float) 0.91749450, (float) 0.91086382, (float) 0.90398929, (float) 0.89687274, (float) 0.88951608,
        (float) 0.88192126, (float) 0.87409034, (float) 0.86602540, (float) 0.85772861, (float) 0.84920218,
        (float) 0.84044840, (float) 0.83146961, (float) 0.82226822, (float) 0.81284668, (float) 0.80320753,
        (float) 0.79335334, (float) 0.78328675, (float) 0.77301045, (float) 0.76252720, (float) 0.75183981,
        (float) 0.74095113, (float) 0.72986407, (float) 0.71858162, (float) 0.70710678, (float) 0.69544264,
        (float) 0.68359230, (float) 0.67155895, (float) 0.65934582, (float) 0.64695615, (float) 0.63439328,
        (float) 0.62166057, (float) 0.60876143, (float) 0.59569930, (float) 0.58247770, (float) 0.56910015,
        (float) 0.55557023, (float) 0.54189158, (float) 0.52806785, (float) 0.51410274, (float) 0.50000000,
        (float) 0.48576339, (float) 0.47139674, (float) 0.45690388, (float) 0.44228869, (float) 0.42755509,
        (float) 0.41270703, (float) 0.39774847, (float) 0.38268343, (float) 0.36751594, (float) 0.35225005,
        (float) 0.33688985, (float) 0.32143947, (float) 0.30590302, (float) 0.29028468, (float) 0.27458862,
        (float) 0.25881905, (float) 0.24298018, (float) 0.22707626, (float) 0.21111155, (float) 0.19509032,
        (float) 0.17901686, (float) 0.16289547, (float) 0.14673047, (float) 0.13052619, (float) 0.11428696,
        (float) 0.09801714, (float) 0.08172107, (float) 0.06540313, (float) 0.04906767, (float) 0.03271908,
        (float) 0.01636173
};

typedef struct supressionnoise{
	
	float inframe[framesize];//带噪时域
	float  outframe[framesize];//滤波后时域
	float  magoutframepre[framesize];//前一帧语音频谱
	float  magoutframe[framesize];//当前帧语音频谱
	float magframe[framesize];//当前帧带噪频谱
	float noise[framesize];//当前帧噪声估计
	float noisepre[framesize];//前一帧噪声
	// noise estimation para
	float p[framesize];
	float ppre[framesize];
	float pmin[framesize];
	float pminpre[framesize];


	float weiner[framesize];
	float snrprior[framesize];
	float snrpost[framesize];
	
}supression;

//连续频谱最小值跟踪
void noiseestimation(supression*self){
	float alpha = 0.7;
	float beta = 0.96;
	float gamma = 0.998;
	for (int  i = 0; i < framesize; i++)
	{    //注意语音增强书上的代码用的功率谱，此处和我的代码统一，开方取幅值谱
		self->p[i] = sqrt(alpha * self->ppre[i] + (1 - alpha)*self->magframe[i] * self->magframe[i]);
		if (self->pminpre[i]<self->p[i])
		{
			self->pmin[i] =gamma * self->pminpre[i] + (1 - gamma) / (1 - beta)*(self->p[i] - beta * self->ppre[i]);
		}
		else
		{
			self->pmin[i] = self->p[i];
		}
		self->noise[i] = self->pmin[i];
		self->ppre[i] = self->p[i];
		self->pminpre[i] = self->pmin[i];

	}
	



	
}
//直接判决法计算先验信噪比
void computesnrprior(supression*self){

	for (int i = 0; i < framesize; i++)

	{
		self->snrpost[i] = self->magframe[i] * self->magframe[i] / (self->noise[i] * self->noise[i]);
		self->snrprior[i] = DD_PR_SNR * (self->magoutframepre[i] * self->magoutframepre[i] / (self->noise[i] * self->noise[i])) + (1 - DD_PR_SNR)*max((self->snrpost[i]-1),0);
		
		

	}

	
}
void weiner(supression *self){
	for (int i = 0; i < framesize; i++)
	{
		self->weiner[i] = self->snrprior[i] / (1 + self->snrprior[i]);
	}
	
	
}

void fft(float *fftin,supression*self,float*real,float*imag){
	
	fftw_complex * in = NULL;
	fftw_complex*out = NULL;
	fftw_plan p;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*framesize);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*framesize);

	for (int i = 0;i <framesize;i++)
	{
		in[i][0] = fftin[i];
		in[i][1] = 0;
	}
	
	p = fftw_plan_dft_1d(framesize, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0;i <framesize;i++)
	{
		real[i]= out[i][0];
		imag[i]= out[i][1];
		self->magframe[i]=sqrt(real[i]*real[i]+imag[i]*imag[i]);
	}
	
	
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
	
	
}
void ifft(float *ifftout,float*real,float*imag){
    fftw_complex * in = NULL;
	fftw_complex*out = NULL;
	fftw_plan p;

	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*framesize);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*framesize);
	for (int i = 0;i <framesize;i++)
	{
		in[i][0] = real[i];
		in[i][1] = imag[i];
	}
	
	p = fftw_plan_dft_1d(framesize, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	for (int i = 0;i <framesize;i++)
	{
		ifftout[i]= out[i][0];
	}
	
	
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
}

int main() {
	unsigned int channels = 0;
	unsigned int sampleRate = 0;
	drwav_uint64 totalSampleCount = 0;
	drwav_int16* pSampleData = drwav_open_and_read_file_s16("C:\\Users\\Administrator\\Desktop\\wiener\\1.wav", &channels, &sampleRate, &totalSampleCount);//默认的就是转成16bit,所以写的时候也要改成16bit
	if (pSampleData == NULL) {
		printf("open failed");
		getchar();
		return 0;
	}
	float*in =(float*)malloc(totalSampleCount*sizeof(float));

	for (drwav_uint64 i = 0; i <totalSampleCount; i++)
	{
		in[i] =pSampleData[i];
	}
	int nframe = totalSampleCount / framesize;

	drwav_uint64 len = nframe * framesize;
	
	
	
	
	supression*self;
	self = (supression *)malloc(sizeof(supression));
	memset(self->snrprior,0,sizeof(float)*framesize);
	memset(self->snrpost,0,sizeof(float)*framesize);

  memset(self->inframe,0,sizeof(float)*framesize);
   memset(self->outframe,0,sizeof(float)*framesize);
   memset(self->magoutframepre,0,sizeof(float)*framesize);
   memset(self->magoutframe,0,sizeof(float)*framesize);
	memset(self->magframe,0,sizeof(float)*framesize);



	memset(self->p, 0, sizeof(float)*framesize);
	memset(self->ppre, 0, sizeof(float)*framesize);
	memset(self->pmin, 0, sizeof(float)*framesize);
	memset(self->pminpre, 0, sizeof(float)*framesize);

	memset(self->noise,0,sizeof(float)*framesize);
	memset(self->noisepre,0,sizeof(float)*framesize);
	memset(self->weiner, 0, sizeof(float)*framesize);


	float *out = (float*)malloc(len*sizeof(float));
	float*pin = &in[0];
	float*pout = &out[0];
	drwav_int32 *outout = (drwav_int32 *)malloc(len * sizeof(drwav_int32));


	float real[framesize] = {0};
	float imag[framesize] = {0};
	 for (int  i = 0; i <nframe; i++)
	 {       

		
	 	for (int m = 0; m < framesize; m++)
		 {
			 self->inframe[m] = pin[m];
		 }
		   
		    fft(self->inframe,self,real,imag);
		
			
		    noiseestimation(self);
		   
		    computesnrprior(self);
		   
		    weiner(self);
			for (int i = 0; i < framesize; i++)
			{
				real[i] *= self->weiner[i];
				imag[i] *= self->weiner[i];
				self->magoutframepre[i] = sqrt(real[i] * real[i]+imag[i]*imag[i]);
			}
		   
		    ifft(self->outframe,real,imag);
		   

		 for (int j = 0; j < framesize; j++)
		 {
			 outout[i*framesize+j] = 10*self->outframe[j];
			 
		 }
	
		 pin += framesize;
		 pout += framesize;
	 }


	drwav_data_format format;
	format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
	format.format = DR_WAVE_FORMAT_PCM;          // <-- Any of the DR_WAVE_FORMAT_* codes.
	format.channels = channels;
	format.sampleRate = sampleRate;
	format.bitsPerSample = 32;
	drwav* pWav = drwav_open_file_write("2.wav", &format);

	drwav_uint64 samplesWritten = drwav_write(pWav, len,outout);
	drwav_free(pSampleData);
	if (samplesWritten > 0)
	{
		printf("success");
	}


	getchar();
     return 0;

}







