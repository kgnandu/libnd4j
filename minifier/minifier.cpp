#include <sys/stat.h>

#ifdef _WIN32
    #include <io.h>
#else
    #include <unistd.h>
#endif

#include "graphopt.h"

int
main(int argc, char *argv[]) {
    GraphOpt opt;

    int err = GraphOpt::optionsWithArgs(opt, argc, argv);
    
    //std::cout << opt << std::endl;
    if (err > 0) {   
        // only help message
        return err;
    }

    if (err < 0) {
        std::cerr << "Wrong parameter list" << std::endl;
        opt.help(argv[0], std::cerr); 
        return err;
    }
    
    for (char option: opt.options()) {
        std::cout << "Option \'" << opt << std::endl;
        switch (option) {
        case 'l':
            std::cout << "Build library" << std::endl;
            break;
        case 'x':
            std::cout << "Build executable" << std::endl;
            break;
        case 'e':
            std::cout << "Link the Graph to executable as Resource" << std::endl;
            break;
        case 'o':
            std::cout << "Output result with name " << opt.outputName() << std::endl;
            break;
        default:
            std::cerr << "Wrong parameter " << option << std::endl;
        }
    }
    
    if (!opt.hasParam('o')) {
        std::cout << "Ouput name is " << opt.outputName() << std::endl;
    }

    for (auto file: opt.files()) {
        // файлы надо проверять на наличие, читаемость и не-нулевую длину
#ifdef _WIN32
        if (_access(file.c_str(), 1) != -1) {
#else
        if (access(file.c_str(), F_OK | R_OK) != -1) {
#endif
#ifdef _WIN32
            struct _stat st;
            _stat(file.c_str(), &st);
#else
            struct stat st;
            stat(file.c_str(), &st);
#endif  
            if (st.st_size != 0)
                std::cout << "File " << file << " exists and can be read" << std::endl;
            else 
                std::cerr << "File " << file << " exists, but has zero size" << std::endl;
        }
        else {
            std::cerr << "File " << file << " does not exists " << std::endl;
        }
    }
    return EXIT_SUCCESS;
}
