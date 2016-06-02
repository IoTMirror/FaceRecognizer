using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Util;
using System.IO;
using System.IO.Compression;

namespace FRLib
{
    public class LBPHUserFaceRecognizer : UserFaceRecognizer
    {
        public string DataDirectory { get; set; }
        public int Radius { get; set; } = 1;
        public int Neighbours { get; set; } = 8;
        public int GridX { get; set; } = 8;
        public int GridY { get; set; } = 8;
        public double Threshold { get; set; } = double.MaxValue;
        public bool Compression { get; set; } = false;

        public LBPHUserFaceRecognizer(string dataDirectory)
        {
            DataDirectory = dataDirectory;
        }

        public int Recognize(Mat image)
        {
            if (image == null) return -1;
            Directory.CreateDirectory(DataDirectory);
            string fileExtension = Compression == true ? "lbph.zip" : "lbph";
            IEnumerable<string> files = Directory.EnumerateFiles(DataDirectory, "*."+fileExtension);
            List<int> users = new List<int>();
            foreach(string file in files)
            {
                string[] nameParts = file.Replace("."+fileExtension, "").Split(new char[] { '/','\\' });
                int userid;
                if (int.TryParse(nameParts[nameParts.Length-1], out userid)) users.Add(userid);
            }
            return Recognize(image, users);
        }

        public int Recognize(Mat image, List<int> usersToCheck)
        {
            if (image == null) return -1;
            int bestUID = -1;
            double bestDistance=0;
            Directory.CreateDirectory(DataDirectory);
            foreach (int uid in usersToCheck)
            {
                if (Compression) DecompressRecognizer(uid);
                LBPHFaceRecognizer lbph = LoadRecognizer(uid);
                if (Compression)
                {
                    RemoveRecognizer(uid);
                }
                try
                {
                    FaceRecognizer.PredictionResult result = lbph.Predict(image);
                    if (result.Distance != double.MaxValue && (bestUID == -1 || result.Distance < bestDistance))
                    {
                        bestUID = result.Label;
                        bestDistance = result.Distance;
                    }
                }
                catch(CvException)
                {
                    //nothing to do
                }
                lbph.Dispose();
            }
            return bestUID;
        }

        public void RemoveUserData(int userId)
        {
            RemoveRecognizer(userId);
            RemoveCompressedRecognizer(userId);
        }

        public void Train(int userId, Mat image)
        {
            Directory.CreateDirectory(DataDirectory);
            if (Compression) DecompressRecognizer(userId);
            LBPHFaceRecognizer lbph = LoadRecognizer(userId);
            VectorOfMat images = new VectorOfMat();
            images.Push(image);
            VectorOfInt labels = new VectorOfInt();
            labels.Push(new int[] { userId });
            lbph.Update(images,labels);
            lbph.Save(DataDirectory + "/" +userId + ".lbph");
            lbph.Dispose();
            if (Compression)
            {
                CompressRecognizer(userId);
                RemoveRecognizer(userId);
            }
        }

        private LBPHFaceRecognizer LoadRecognizer(int userId)
        {
            LBPHFaceRecognizer lbph = new LBPHFaceRecognizer(Radius, Neighbours, GridX, GridY, Threshold);
            try
            {
                lbph.Load(DataDirectory + "/" + userId + ".lbph");
            }
            catch (CvException)
            {
                //nothing to do - new Face Recognizer
            }
            return lbph;
        }

        private void RemoveRecognizer(int userId)
        {
            Directory.CreateDirectory(DataDirectory);
            File.Delete(DataDirectory + "/" + userId + ".lbph");
        }

        private void RemoveCompressedRecognizer(int userId)
        {
            Directory.CreateDirectory(DataDirectory);
            File.Delete(DataDirectory + "/" + userId + ".lbph.zip");
        }

        private void CompressRecognizer(int userId)
        {
            FileStream fstream = null;
            try
            {
                fstream = new FileStream(DataDirectory + "/" + userId + ".lbph.zip", FileMode.OpenOrCreate);
            }
            catch (Exception)
            {
                return;
            }
            ZipArchive zip = null;
            try
            {
                zip = new ZipArchive(fstream, ZipArchiveMode.Create, false);
                zip.CreateEntryFromFile(DataDirectory + "/" + userId + ".lbph", "" + userId + ".lbph");
                zip.Dispose();
            }
            catch(Exception)
            {
                fstream.Close();
            }
        }

        private void DecompressRecognizer(int userId)
        {
            try
            {
                ZipFile.ExtractToDirectory(DataDirectory + "/" + userId + ".lbph.zip", DataDirectory+"/");
            }
            catch(Exception)
            {
                //nothing to do
            }
        }
    }
}
