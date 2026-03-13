################################################################################
# Vihaan RMSF Plugin for VMD
#
# Native Tcl/Tk plugin to compute per-atom RMSF (Root Mean Square Fluctuation)
# for the currently loaded top molecule trajectory in VMD.
#
# Monolithic-System Alignment:
# - Entire plugin is implemented in a single Tcl file.
# - All state and procedures are encapsulated in one namespace.
# - No cross-module runtime dependencies are introduced.
#
# Core Workflow:
# 1) Build atom selection from GUI text input.
# 2) First frame pass: compute per-atom average coordinate (x,y,z).
# 3) Second frame pass: accumulate squared displacement from average.
# 4) RMSF = sqrt( mean( squared displacement ) ).
# 5) Optional conversion to theoretical B-factor:
#      B = (8 * pi^2 / 3) * (RMSF^2)
# 6) Write final values into VMD "user" field for coloring/visual analysis.
#
# CRITICAL implementation requirement honored:
# - Uses "$sel frame $i" and "$sel update" in both frame loops.
# - Always calls "$sel delete" before procedure exit.
#
################################################################################

namespace eval ::VihaanRMSF:: {
    # -------------------------------------------------------------------------
    # Namespace variables (GUI-bound state)
    # -------------------------------------------------------------------------
    # Atom selection string shown in GUI entry box.
    variable selection_text "protein and name CA"

    # Checkbox state:
    #   0 -> store RMSF in user field
    #   1 -> store theoretical B-factor in user field
    variable calc_bfactor 0

    # Window path for plugin GUI.
    variable w ".vihaan_rmsf_gui"

    # -------------------------------------------------------------------------
    # Procedure: log
    # Centralized logger for consistent plugin output in VMD console.
    # -------------------------------------------------------------------------
    proc log {level message} {
        puts "\[VihaanRMSF\]\[$level\] $message"
    }

    # -------------------------------------------------------------------------
    # Procedure: gui
    # Creates or raises the plugin window.
    # -------------------------------------------------------------------------
    proc gui {} {
        variable w
        variable selection_text
        variable calc_bfactor

        # If the GUI already exists, raise it and stop.
        if {[winfo exists $w]} {
            wm deiconify $w
            raise $w
            focus $w
            return
        }

        # Create top-level plugin window.
        toplevel $w
        wm title $w "Vihaan RMSF Calculator"
        wm resizable $w 1 0

        # Keep layout simple, robust, and monolithic.
        frame $w.main -padx 10 -pady 10
        pack $w.main -side top -fill both -expand 1

        # -------------------------
        # Selection input row
        # -------------------------
        label $w.main.sel_lbl -text "Atom Selection:"
        entry $w.main.sel_ent \
            -textvariable ::VihaanRMSF::selection_text \
            -width 40

        grid $w.main.sel_lbl -row 0 -column 0 -sticky w -padx 4 -pady 4
        grid $w.main.sel_ent -row 0 -column 1 -sticky ew -padx 4 -pady 4
        grid columnconfigure $w.main 1 -weight 1

        # -------------------------
        # Optional B-factor toggle
        # -------------------------
        checkbutton $w.main.bfac_chk \
            -text "Signature: Convert to Theoretical B-factor" \
            -variable ::VihaanRMSF::calc_bfactor

        grid $w.main.bfac_chk -row 1 -column 0 -columnspan 2 -sticky w -padx 4 -pady 6

        # -------------------------
        # Execute button
        # -------------------------
        button $w.main.calc_btn \
            -text "Calculate" \
            -command ::VihaanRMSF::calculate

        grid $w.main.calc_btn -row 2 -column 0 -columnspan 2 -sticky ew -padx 4 -pady 8

        # Helpful status note.
        label $w.main.note \
            -text "Results are written to the per-atom 'user' field for coloring." \
            -fg "#333333"
        grid $w.main.note -row 3 -column 0 -columnspan 2 -sticky w -padx 4 -pady 2
    }

    # -------------------------------------------------------------------------
    # Procedure: calculate
    # Performs two-pass RMSF computation on top molecule trajectory.
    # -------------------------------------------------------------------------
    proc calculate {} {
        variable selection_text
        variable calc_bfactor

        # Identify top molecule in VMD.
        set molid [molinfo top]
        if {$molid < 0} {
            tk_messageBox -icon warning -type ok -title "Vihaan RMSF" \
                -message "No top molecule is loaded in VMD."
            return
        }

        # Total number of frames in trajectory.
        set numframes [molinfo $molid get numframes]
        if {$numframes < 1} {
            tk_messageBox -icon warning -type ok -title "Vihaan RMSF" \
                -message "Top molecule has no trajectory frames."
            return
        }

        # Build selection once. We retarget frame index during loops.
        set sel [atomselect $molid $selection_text]

        # Guard against invalid/empty selection.
        set natoms [$sel num]
        if {$natoms < 1} {
            $sel delete
            tk_messageBox -icon warning -type ok -title "Vihaan RMSF" \
                -message "Selection matched zero atoms: $selection_text"
            return
        }

        # Wrap heavy work in catch so we can always delete selection and avoid leaks.
        if {[catch {
            ::VihaanRMSF::log INFO "Starting RMSF for $natoms atoms across $numframes frames"
            ::VihaanRMSF::log INFO "Selection: $selection_text"

            # =================================================================
            # PASS 1: Compute average coordinates per atom
            # =================================================================
            # We accumulate x/y/z sums independently to keep code explicit.
            set sumx [lrepeat $natoms 0.0]
            set sumy [lrepeat $natoms 0.0]
            set sumz [lrepeat $natoms 0.0]

            for {set i 0} {$i < $numframes} {incr i} {
                # CRITICAL: required by specification
                $sel frame $i
                $sel update

                # coords is a list of natoms items; each item is "{x y z}".
                set coords [$sel get {x y z}]

                for {set j 0} {$j < $natoms} {incr j} {
                    lassign [lindex $coords $j] x y z
                    lset sumx $j [expr {[lindex $sumx $j] + $x}]
                    lset sumy $j [expr {[lindex $sumy $j] + $y}]
                    lset sumz $j [expr {[lindex $sumz $j] + $z}]
                }
            }

            # Convert sums to averages.
            set avgx {}
            set avgy {}
            set avgz {}
            set invframes [expr {1.0 / double($numframes)}]

            for {set j 0} {$j < $natoms} {incr j} {
                lappend avgx [expr {[lindex $sumx $j] * $invframes}]
                lappend avgy [expr {[lindex $sumy $j] * $invframes}]
                lappend avgz [expr {[lindex $sumz $j] * $invframes}]
            }

            # =================================================================
            # PASS 2: Compute RMSF from squared displacement to average
            # =================================================================
            set sumsq [lrepeat $natoms 0.0]

            for {set i 0} {$i < $numframes} {incr i} {
                # CRITICAL: required by specification
                $sel frame $i
                $sel update

                set coords [$sel get {x y z}]

                for {set j 0} {$j < $natoms} {incr j} {
                    lassign [lindex $coords $j] x y z

                    set dx [expr {$x - [lindex $avgx $j]}]
                    set dy [expr {$y - [lindex $avgy $j]}]
                    set dz [expr {$z - [lindex $avgz $j]}]

                    # Squared displacement in 3D.
                    set d2 [expr {$dx*$dx + $dy*$dy + $dz*$dz}]
                    lset sumsq $j [expr {[lindex $sumsq $j] + $d2}]
                }
            }

            # =================================================================
            # Final values: RMSF or theoretical B-factor
            # =================================================================
            set values {}
            set bscale [expr {(8.0 * acos(-1.0) * acos(-1.0)) / 3.0}]
            set minval ""
            set maxval ""
            set sumval 0.0

            for {set j 0} {$j < $natoms} {incr j} {
                set mean_d2 [expr {[lindex $sumsq $j] * $invframes}]
                set rmsf [expr {sqrt($mean_d2)}]

                if {$calc_bfactor} {
                    # Theoretical B-factor conversion.
                    set outval [expr {$bscale * $rmsf * $rmsf}]
                } else {
                    set outval $rmsf
                }

                if {$j == 0} {
                    set minval $outval
                    set maxval $outval
                } else {
                    if {$outval < $minval} {
                        set minval $outval
                    }
                    if {$outval > $maxval} {
                        set maxval $outval
                    }
                }
                set sumval [expr {$sumval + $outval}]

                lappend values $outval
            }

            set meanval [expr {$sumval / double($natoms)}]

            # Apply computed values to VMD's user field for coloring.
            $sel set user $values

            if {$calc_bfactor} {
                set mode_label "B-factor"
            } else {
                set mode_label "RMSF"
            }

            ::VihaanRMSF::log INFO "Completed calculation. Mode=$mode_label Atoms=$natoms Frames=$numframes Min=$minval Max=$maxval Mean=$meanval"

            if {$calc_bfactor} {
                ::VihaanRMSF::log INFO "Stored theoretical B-factor in 'user' field."
                tk_messageBox -icon info -type ok -title "Vihaan RMSF" \
                    -message "Calculation complete. Stored theoretical B-factor in user field."
            } else {
                ::VihaanRMSF::log INFO "Stored RMSF in 'user' field."
                tk_messageBox -icon info -type ok -title "Vihaan RMSF" \
                    -message "Calculation complete. Stored RMSF in user field."
            }
        } err]} {
            # Report failures but continue cleanup below.
            ::VihaanRMSF::log ERROR "RMSF calculation failed: $err"
            tk_messageBox -icon error -type ok -title "Vihaan RMSF" \
                -message "RMSF calculation failed:\n$err"
        }

        # CRITICAL: always delete selection to prevent memory leaks.
        $sel delete
    }
}

################################################################################
# VMD Extension Menu Registration
#
# Registers plugin into VMD menu tree:
# Extensions -> Vihaan RMSF Calculator
################################################################################
menu tk register "Vihaan RMSF Calculator" ::VihaanRMSF::gui
