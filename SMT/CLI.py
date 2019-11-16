import utils
import IBM_Model
import sims

def translate(model_DE=None,model_ED=None):
    
    choice = int(input("""\nDo you want to translate from :
          Dutch to English (Enter 1)
          English to Dutch (Enter 2)\n"""))
    model = None
    if(choice==1):
        if(model_DE==None):
            model_DE = utils.get_modelDE()
        model = model_DE
    elif(choice==2):
        if(model_ED==None):
            model_ED = utils.get_modelED()
        model = model_ED
    else:
        print("Enter either a 1 or 2. ")
        
    num = int(input("Number of test documents you want to translate : "))
    checks = input("Do you want to compute (average) cosine similarities and (average) Jaccard coefficient of these documents?(Y/N) ")
    checks = (checks=='Y') or (checks=='y')
    cosines = []
    jaccards = []
    for i in range(num):
        # Dutch to English
        
        testPath = input("Specify path of Test Document(to be translated) : ")
        doc = utils.get_data(testPath)
        translated_doc = model.translate(doc)
        if(checks):
            origPath = input("Specify path of Expected translated Document : ")
            origDoc = utils.get_data(origPath)
            cosines.append(sims.cosine_similarity(origDoc,translated_doc))
            jaccards.append(sims.jaccard_coefficient(origDoc,translated_doc))
        with open("Document " + str(i+1)+'.txt', 'w') as f:
            for line in translated_doc:
                f.write("%s\n" % line)
        
        print("Document "+str(i+1)+" translated successfully.")
        
    if(checks):
        for i in range(num):
            print("Cosine similarity for doc "+str(i+1)+": "+str(cosines[i]))
            print("Jaccard coefficient for doc "+str(i+1)+": "+str(jaccards[i]))

        print("Average cosine similarity: "+str(sum(cosines)/num))
        print("Average Jaccard coefficient: "+str(sum(jaccards)/num))

loop = True

while loop:
    print("\nChoose one of the following options :",end='')
    print("""
        Enter 1 for training a corpus
        Enter 2 for translating a document.
        Enter 3 for computing Cosine Similarity for 2 documents.
        Enter 4 for computing Jaccard coefficient for 2 documents.
        Enter 0 for exiting the console.
    \t""")

    choice = int(input("Input : \t"))

    if choice == 1:  # Training a document

        modelChoice = int(input("Which IBM model do you want to implement - 1 or 2  : "))

        # C:\Users\Rohit Bohra\Desktop\IR Assignment\Final\data
        engPath = input("Path of English Document : ")
        englishFile = utils.get_data(engPath)
        dutchPath = input("Path of Dutch Document : ")
        dutchFile = utils.get_data(dutchPath)

        assert len(englishFile) == len(dutchFile), "Length of EnglishFile and DutchFile needs to be same"
        model_DE = None
        model_ED = None
        
        while(True):
        
            if modelChoice == 1:
                model_DE = IBM_Model.IBM1()  # For Dutch to English Translation
                print("Training for Dutch to English..")
                model_DE.run_iter(englishFile[0:10000], dutchFile[0:10000], NumIter=7)
                # After 7 iterations, the log-likehood function doesn't change much

                model_ED = IBM_Model.IBM1()  # For English to Dutch Translation
                print("Training for English to Dutch..")
                model_ED.run_iter(englishFile[0:10000], dutchFile[0:10000], NumIter=7)
                break

            if modelChoice == 2:
                model_DE = IBM_Model.IBM2()  # For Dutch to English Translation
                print("Training for Dutch to English..")
                model_DE.run_iter(englishFile[0:10000], dutchFile[0:10000], NumIter=7)
                # After 7 iterations, the log-likehood function doesn't change much

                model_ED = IBM_Model.IBM1()  # For English to Dutch Translation
                print("Training for English to Dutch..")
                model_ED.run_iter(englishFile[0:10000], dutchFile[0:10000], NumIter=7)
                break
            else:
                modelChoice = int(input("Enter either a 1 or 2: "))
                
        check = input("Model is trained!\nDo you want to translate further?(Y/N) ")
        if(check=='Y'or check=='y'):
            translate(model_DE,model_ED)
        
        
    elif choice == 2:  # Translating a document
        translate()
    elif choice == 3:  # Computing Cosine Similarity
        path_doc1 = input("Path for Document-1 : ")
        path_doc2 = input("Path for Document-2 : ")

        doc1 = utils.get_data(path_doc1)
        doc2 = utils.get_data(path_doc2)

        print("Cosine Similarity : ", sims.cosine_similarity(doc1, doc2))

    elif choice == 4:  # Computing Jaccard coefficient
        path_doc1 = input("Path for Document-1 : ")
        path_doc2 = input("Path for Document-2 : ")

        doc1 = utils.get_data(path_doc1)
        doc2 = utils.get_data(path_doc2)

        print("Jaccard coefficient coefficient : ", sims.jaccard_coefficient(doc1, doc2))

    elif choice == 0:
        print("Exiting Console\n")
        break
    else:
        print("Please enter a valid choice\n")
        
    check = input("Do you want to continue?(Y/N) ")
    loop = (check=='Y'or check=='y')
    
