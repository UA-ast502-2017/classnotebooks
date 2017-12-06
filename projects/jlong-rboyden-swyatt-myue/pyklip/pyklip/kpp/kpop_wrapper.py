__author__ = 'JB'

def kpop_wrapper(inputDir,obj_list,spectrum_list = None,outputDir = None, mute_error = True):
    '''
    Iterate over:
        - a list reduction objects (inherited from kppSuperClass)
        - a list of reduction spectra
        - all the files matching the filename description in each object.
    
    Args:
        inputDir: Directory from which the filenames will be read.
        obj_list: List of reduction objects inherited from kppSuperClass
        spectrum_list: List of spectrum to be used for the reductions.
                        For e.g.: ["t600g100nc","t1300g100f2",np.ones(37),"host_star_spec"]
                        Each spectrum can be a spectrum name (string) or an array
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
        outputDir: Output directory for the processed files.
        mute_error: If True (default), don't crash if one reduction failed. Just go to the next one.

    Return:
        err_list: List of errors from the failed reduction.
    '''

    err_list = []
    for obj in obj_list:
        iterating = True
        while iterating:
            if not mute_error:
                iterating = obj.initialize(inputDir=inputDir,outputDir=outputDir)
                if obj.spectrum_iter_available() and spectrum_list is not None:
                    for spec_path in spectrum_list:
                        obj.init_new_spectrum(spec_path)
                        run(obj)
                else:
                    run(obj)
            else:
                try:
                    iterating = obj.initialize(inputDir=inputDir,outputDir=outputDir)

                    if obj.spectrum_iter_available() and spectrum_list is not None:
                        for spec_path in spectrum_list:
                            try:
                                obj.init_new_spectrum(spec_path)
                                run(obj)
                            except Exception as myErr:
                                err_list.append(myErr)
                                print("//!\\\\ "+obj.filename+"with spectrum "+spec_path+" in "+inputDir+" raised an Error.")
                    else:
                        try:
                            run(obj)
                        except Exception as myErr:
                            err_list.append(myErr)
                            print("//!\\\\ "+obj.filename+" in "+inputDir+" raised an Error.")
                except Exception as myErr:
                    err_list.append(myErr)
                    iterating = False
                    print("//!\\\\ "+obj.filename+" could NOT initialize in "+inputDir+". raised an Error.")


    return err_list


def run(obj):
    if not obj.check_existence():
        map = obj.calculate()
        obj.save()
    else:
        pass
        #map = obj.load()

    return None