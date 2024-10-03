## Description of the dataset

The experiment recruited 12 participants, comprising 6 males and 6 females, aged between 20 and 27 years. The details are shown in the table below. Prior to the data collection process, participants were briefly introduced to the system's working principle and given a demonstration of the writing area. Beyond this, no restrictions were placed on the writing speed, stroke order, or habits of the volunteers, ensuring a natural writing style. Words from the given vocabulary list appeared randomly in the program, with each volunteer inputting words as prompted. Each user collected an average of about 1200 samples, totaling 15488 samples. The dataset is available in this [Link](http://example.com).

| Dataset Folder | Volunteer ID | Gender | Age | Handedness | Number of Samples | Reserved |
|----------------|--------------|--------|-----|------------|-------------------|----------|
| datas1         | -            | -      | -   | -          | -                 |  Access not supported |
| datas2         | 001          | Male   | 26  | Right      | 1212              | -        |
| datas3         | 002          | Male   | 26  | Right      | 1202              | -        |
| datas4         | 003          | Male   | 25  | Right      | 1197              | -        |
| datas5         | 004          | Male   | 26  | Right      |  613              | -        |
| datas6         | 005          | Male   | 22  | Right      | 1621              | -        |
| datas7         | 006          | Female | 23  | Right      | 1271              | -        |
| datas8         | 007          | Female | 22  | Right      | 1299              | -        |
| datas9         | 008          | Male   | 22  | Right      |  750              | -        |
| datas10        | 009          | Female | 23  | Right      | 1261              | -        |
| datas11        | 010          | Female | 22  | Right      | 1278              | -        |
| datas12        | 011          | Male   | 22  | Right      | 1324              | -        |
| datas13        | 012          | Male   | 22  | Right      | 1256              | -        |
| datas14        | 013          | Male   | 21  | Right      | 1192              | -        |

## Data Collection System

To achieve efficient, high-quality dataset construction, we designed a synchronized data collection system combining millimeter-wave radar and a Leap Motion controller. This system connects to a PC via USB interface to run the data collection application, as illustrated in the Figure below. The Leap Motion accurately tracks hand joint data, enabling automatic and precise truncation of collected data length. The tracked motion trajectory of the index fingertip is considered ground truth, facilitating subsequent data quality selection and alignment processing.

<p align="center">  
    <img src="../img/fig7.png" alt=" " width="500" />  
</p>  
<p align="center"></p>

## Dataset Format

Collected data is stored in .npy file format. The data naming format is: $\rm data\_{yyyymmdd\_hhmmss}.npy$, for example, $\rm'data\_20240508\_210234.npy'$. The data shape is $T*163$, where $T$ is the data frame length. Here, data[:,:128] represents micro-Doppler time features, data[:, 128:160] represents range-time features, and data[:, 160:162] contains x-z coordinate position information of the right index fingertip collected by Leap Motion. data[:, 163] is reserved. Data visualization is shown in the Figure below. This study only utilizes micro-Doppler time features, range-time features and right index fingertip coordinate position information are not used, reserving more candidate information for future research.

<p align="center">  
    <img src="../img/fig9.png" alt=" " width="600" />  
</p>  
<p align="center"></p>

